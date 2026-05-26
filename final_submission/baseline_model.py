"""
Baseline Model Training and Evaluation

This script exactly follows benchmark.ipynb:
1. Model architecture matches benchmark.ipynb
2. Uses TFMA for 13 metrics evaluation
3. Adds 5 extended fairness metrics

Based on benchmark.ipynb baseline model.
"""

import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
from google.protobuf import text_format
import json

RANDOM_STATE = 200
BATCH_SIZE = 100
EPOCHS = 10

LABEL_KEY = "EMPLOYED"
SENSITIVE_ATTRIBUTE_KEY = "SEX"
BANNED_FEATURES = ["RELP"]
PREDICTION_KEY = "PRED"
SENSITIVE_ATTRIBUTE_VALUES = {1.0: "Male", 2.0: "Female"}


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_baseline_model(features):
    """Build baseline model exactly as in benchmark.ipynb."""
    inputs = {}
    for name, column in features.items():
        if name != LABEL_KEY:
            inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=tf.float64)

    def stack_dict(inputs, fun=tf.stack):
        values = []
        for key in sorted(inputs.keys()):
            values.append(tf.cast(inputs[key], tf.float64))
        return fun(values, axis=-1)

    x = stack_dict(inputs, fun=tf.concat)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(stack_dict(dict(features)))

    x = normalizer(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs)


def _format_slice_name(slice_name):
    """Format TFMA slice name."""
    if slice_name == ():
        return "Overall"
    parts = []
    for key, value in slice_name:
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _format_tfma_value(value):
    """Format TFMA metric value."""
    if isinstance(value, dict):
        if "doubleValue" in value:
            return f"{value['doubleValue']:.4f}"
        if "value" in value and isinstance(value["value"], (int, float)):
            return f"{value['value']:.4f}"
        if "boundedValue" in value and isinstance(value["boundedValue"], dict):
            bounded = value["boundedValue"]
            lower = bounded.get("lowerBound")
            upper = bounded.get("upperBound")
            mean = bounded.get("value")
            return f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_tfma_text_summary(eval_result, title):
    """Print TFMA evaluation summary."""
    print(f"\nTFMA Text Summary: {title}")
    metrics_by_slice = eval_result.get_metrics_for_all_slices()
    for slice_name in sorted(metrics_by_slice.keys(), key=str):
        print(f"\nSlice: {_format_slice_name(slice_name)}")
        metrics = metrics_by_slice[slice_name]
        for metric_name in sorted(metrics.keys()):
            print(f"  {metric_name}: {_format_tfma_value(metrics[metric_name])}")


def compute_extended_fairness_metrics(df, pred_col=PREDICTION_KEY, label_col=LABEL_KEY,
                                     sensitive_col=SENSITIVE_ATTRIBUTE_KEY, threshold=0.5):
    """Compute 5 extended fairness metrics (only improved ones)."""
    y_pred_proba = df[pred_col].values
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_true = df[label_col].values
    sensitive = df[sensitive_col].values

    female_mask = (sensitive == 'Female')
    male_mask = (sensitive == 'Male')

    y_true_f = y_true[female_mask]
    y_pred_f = y_pred[female_mask]
    y_pred_proba_f = y_pred_proba[female_mask]

    y_true_m = y_true[male_mask]
    y_pred_m = y_pred[male_mask]
    y_pred_proba_m = y_pred_proba[male_mask]

    metrics = {}

    # 1. Disparate Impact (80% Rule)
    pos_rate_f = (y_pred_f == 1).mean()
    pos_rate_m = (y_pred_m == 1).mean()
    ratio = pos_rate_f / pos_rate_m if pos_rate_m > 0 else 0
    metrics['disparate_impact'] = {
        'female_positive_rate': float(pos_rate_f),
        'male_positive_rate': float(pos_rate_m),
        'ratio': float(ratio),
        'fair': 0.8 <= ratio <= 1.25,
    }

    # 2. Average Odds
    tpr_f = (y_pred_f[y_true_f == 1] == 1).mean() if (y_true_f == 1).sum() > 0 else 0
    tpr_m = (y_pred_m[y_true_m == 1] == 1).mean() if (y_true_m == 1).sum() > 0 else 0
    fpr_f = (y_pred_f[y_true_f == 0] == 1).mean() if (y_true_f == 0).sum() > 0 else 0
    fpr_m = (y_pred_m[y_true_m == 0] == 1).mean() if (y_true_m == 0).sum() > 0 else 0
    average_odds_diff = (abs(tpr_f - tpr_m) + abs(fpr_f - fpr_m)) / 2
    metrics['average_odds'] = {
        'female_tpr': float(tpr_f),
        'male_tpr': float(tpr_m),
        'female_fpr': float(fpr_f),
        'male_fpr': float(fpr_m),
        'difference': float(average_odds_diff),
        'fair': average_odds_diff < 0.1,
    }

    # 3. Predictive Parity
    ppv_f = (y_true_f[y_pred_f == 1] == 1).mean() if (y_pred_f == 1).sum() > 0 else 0
    ppv_m = (y_true_m[y_pred_m == 1] == 1).mean() if (y_pred_m == 1).sum() > 0 else 0
    predictive_parity_diff = abs(ppv_f - ppv_m)
    metrics['predictive_parity'] = {
        'female_precision': float(ppv_f),
        'male_precision': float(ppv_m),
        'difference': float(predictive_parity_diff),
        'fair': predictive_parity_diff < 0.1,
    }

    # 4. NPV Equality (Conditional Use Accuracy)
    tn_f = ((y_true_f == 0) & (y_pred_f == 0)).sum()
    fn_f = ((y_true_f == 1) & (y_pred_f == 0)).sum()
    npv_f = tn_f / (tn_f + fn_f) if (tn_f + fn_f) > 0 else 0

    tn_m = ((y_true_m == 0) & (y_pred_m == 0)).sum()
    fn_m = ((y_true_m == 1) & (y_pred_m == 0)).sum()
    npv_m = tn_m / (tn_m + fn_m) if (tn_m + fn_m) > 0 else 0

    npv_diff = abs(npv_f - npv_m)
    metrics['npv_equality'] = {
        'female_npv': float(npv_f),
        'male_npv': float(npv_m),
        'difference': float(npv_diff),
        'fair': npv_diff < 0.1,
    }

    # 5. Performance Gaps (AUC only)
    from sklearn.metrics import roc_auc_score
    auc_f = roc_auc_score(y_true_f, y_pred_proba_f) if len(np.unique(y_true_f)) > 1 else 0
    auc_m = roc_auc_score(y_true_m, y_pred_proba_m) if len(np.unique(y_true_m)) > 1 else 0
    auc_gap = abs(auc_f - auc_m)

    metrics['performance_gaps_auc'] = {
        'female_auc': float(auc_f),
        'male_auc': float(auc_m),
        'auc_gap': float(auc_gap),
        'fair': auc_gap < 0.05,
    }

    # Summary
    fair_count = sum([
        metrics['disparate_impact']['fair'],
        metrics['average_odds']['fair'],
        metrics['predictive_parity']['fair'],
        metrics['npv_equality']['fair'],
        metrics['performance_gaps_auc']['fair'],
    ])

    metrics['summary'] = {
        'total_metrics': 5,
        'fair_metrics': int(fair_count),
        'fairness_score': float(fair_count / 5),
    }

    return metrics

def print_extended_metrics(metrics, title="Extended Fairness Metrics"):
    """Print extended fairness metrics."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

    print("\n1. DISPARATE IMPACT (80% Rule)")
    print(f"   Female Rate: {metrics['disparate_impact']['female_positive_rate']:.4f}")
    print(f"   Male Rate:   {metrics['disparate_impact']['male_positive_rate']:.4f}")
    print(f"   Ratio:       {metrics['disparate_impact']['ratio']:.4f}")
    print(f"   Fair:        {'✓' if metrics['disparate_impact']['fair'] else '✗'}")

    print("\n2. AVERAGE ODDS")
    print(f"   Diff: {metrics['average_odds']['difference']:.4f}")
    print(f"   Fair: {'✓' if metrics['average_odds']['fair'] else '✗'}")

    print("\n3. PREDICTIVE PARITY")
    print(f"   Female Prec: {metrics['predictive_parity']['female_precision']:.4f}")
    print(f"   Male Prec:   {metrics['predictive_parity']['male_precision']:.4f}")
    print(f"   Diff:        {metrics['predictive_parity']['difference']:.4f}")
    print(f"   Fair:        {'✓' if metrics['predictive_parity']['fair'] else '✗'}")

    print("\n4. NPV EQUALITY (Conditional Use Accuracy)")
    print(f"   Female NPV: {metrics['npv_equality']['female_npv']:.4f}")
    print(f"   Male NPV:   {metrics['npv_equality']['male_npv']:.4f}")
    print(f"   Diff:       {metrics['npv_equality']['difference']:.4f}")
    print(f"   Fair:       {'✓' if metrics['npv_equality']['fair'] else '✗'}")

    print("\n5. PERFORMANCE GAPS (AUC)")
    print(f"   Female AUC: {metrics['performance_gaps_auc']['female_auc']:.4f}")
    print(f"   Male AUC:   {metrics['performance_gaps_auc']['male_auc']:.4f}")
    print(f"   AUC Gap:    {metrics['performance_gaps_auc']['auc_gap']:.4f}")
    print(f"   Fair:       {'✓' if metrics['performance_gaps_auc']['fair'] else '✗'}")

    print("\n" + "="*80)
    print(f"  SUMMARY: {metrics['summary']['fair_metrics']}/{metrics['summary']['total_metrics']} Fair")
    print("="*80)

def main():
    print("="*80)
    print("  Baseline Model (Following benchmark.ipynb)")
    print("="*80)

    set_seeds(RANDOM_STATE)

    # Load data
    acs_df = pd.read_csv("acsemployment_2018_ca_tx.csv")
    acs_df[LABEL_KEY] = acs_df[LABEL_KEY].astype(int)

    # Split data (exactly as benchmark.ipynb)
    acs_train_df = acs_df.sample(frac=0.8, random_state=RANDOM_STATE)
    acs_test_df = acs_df.drop(acs_train_df.index).sample(frac=1.0)  # No random_state here!

    # Prepare features
    features = acs_df.copy()
    features.pop(LABEL_KEY)
    for banned in BANNED_FEATURES:
        if banned in features:
            features.pop(banned)
    if SENSITIVE_ATTRIBUTE_KEY in features:
        features.pop(SENSITIVE_ATTRIBUTE_KEY)

    # Build and compile model
    print("\nBuilding baseline model...")
    base_model = build_baseline_model(features)
    base_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Prepare datasets (following benchmark.ipynb)
    def dataframe_to_dataset(dataframe):
        dataframe = dataframe.copy()
        labels = dataframe.pop(LABEL_KEY)
        dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        return dataset

    acs_train_ds = dataframe_to_dataset(acs_train_df).batch(BATCH_SIZE)
    acs_test_ds = dataframe_to_dataset(acs_test_df).batch(BATCH_SIZE)

    # Train model
    print("\nTraining baseline model...")
    history = base_model.fit(
        acs_train_ds,
        epochs=EPOCHS,
        validation_data=acs_test_ds,
        verbose=0
    )

    # Print training progress
    for epoch in [0, 4, 9, 14, 19, 24]:
        if epoch < len(history.history['loss']):
            print(f"Epoch {epoch+1:02d}  "
                  f"loss={history.history['loss'][epoch]:.4f} "
                  f"auc={history.history['auc'][epoch]:.4f}  "
                  f"val_loss={history.history['val_loss'][epoch]:.4f} "
                  f"val_auc={history.history['val_auc'][epoch]:.4f}")

    # Predict
    print("\nGenerating predictions...")
    test_ds_for_pred = tf.data.Dataset.from_tensor_slices(dict(acs_test_df[list(features.keys())])).batch(BATCH_SIZE)
    base_model_predictions = base_model.predict(test_ds_for_pred, verbose=0).flatten()

    # Prepare for TFMA evaluation
    base_model_analysis = acs_test_df.copy()
    base_model_analysis[SENSITIVE_ATTRIBUTE_KEY].replace(SENSITIVE_ATTRIBUTE_VALUES, inplace=True)
    base_model_analysis[PREDICTION_KEY] = base_model_predictions

    # TFMA evaluation config
    eval_config_pbtxt = """
      model_specs {
        prediction_key: "%s"
        label_key: "%s" }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        metrics { class_name: "BinaryAccuracy" }
        metrics { class_name: "AUC" }
        metrics { class_name: "ConfusionMatrixPlot" }
        metrics {
          class_name: "FairnessIndicators"
          config: '{"thresholds": [0.50]}'
        }
      }
      slicing_specs {
        feature_keys: "%s"
      }
      slicing_specs {}
    """ % (PREDICTION_KEY, LABEL_KEY, SENSITIVE_ATTRIBUTE_KEY)
    eval_config = text_format.Parse(eval_config_pbtxt, tfma.EvalConfig())

    # Run TFMA
    print("\nRunning TFMA evaluation...")
    base_model_eval_result = tfma.analyze_raw_data(base_model_analysis, eval_config)
    print_tfma_text_summary(base_model_eval_result, "Baseline Model")

    # Compute extended fairness metrics
    print("\n" + "="*80)
    print("  EXTENDED FAIRNESS METRICS")
    print("="*80)
    extended_metrics = compute_extended_fairness_metrics(base_model_analysis)
    print_extended_metrics(extended_metrics, "Baseline Model - Extended Metrics")

    # Save results
    base_model_analysis.to_csv('baseline_predictions.csv', index=False)

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        return obj

    with open('baseline_extended_metrics.json', 'w') as f:
        json.dump(convert_types(extended_metrics), f, indent=2)

    print("\nResults saved:")
    print("  - baseline_predictions.csv")
    print("  - baseline_extended_metrics.json")


if __name__ == "__main__":
    main()
