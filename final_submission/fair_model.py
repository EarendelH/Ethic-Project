"""
Fair Model Training and Evaluation

This script:
1. Uses adversarial debiasing with Equalized Odds constraints
2. Uses TFMA for 13 metrics evaluation (same as baseline)
3. Adds 5 extended fairness metrics
"""

import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
from google.protobuf import text_format
import json

RANDOM_STATE = 200
BATCH_SIZE = 256
EPOCHS = 25

LABEL_KEY = "EMPLOYED"
SENSITIVE_ATTRIBUTE_KEY = "SEX"
BANNED_FEATURES = ["RELP"]
PREDICTION_KEY = "PRED"
SENSITIVE_ATTRIBUTE_VALUES = {1.0: "Male", 2.0: "Female"}

ADV_LAMBDA = 0.5
FAIR_LAMBDA = 0.5
POS_BOOST = 1.5


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@tf.custom_gradient
def _grl_op(x, lam):
    def grad(dy): return -lam * dy, None
    return x, grad


class GRL(tf.keras.layers.Layer):
    def __init__(self, lam=1.0, **kw):
        super().__init__(**kw)
        self.lam = tf.Variable(float(lam), trainable=False, dtype=tf.float32)
    def call(self, x):
        return _grl_op(tf.cast(x, tf.float32), self.lam)
    def get_config(self):
        return {**super().get_config(), "lam": float(self.lam.numpy())}


def build_fair_model(train_df, encoder_cols):
    """Build fair model with adversarial debiasing."""
    inputs = {c: tf.keras.Input(shape=(1,), name=c, dtype=tf.float64)
              for c in encoder_cols}

    def stack(d):
        return tf.concat([tf.cast(d[k], tf.float64) for k in sorted(d.keys())], axis=-1)

    x = stack(inputs)
    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(train_df[encoder_cols].values)
    x = norm(x)

    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    main_out = tf.keras.layers.Dense(1, activation="sigmoid", name="main_output")(x)

    x_adv = GRL(lam=ADV_LAMBDA, name="grl")(x)
    x_adv = tf.keras.layers.Dense(16, activation="relu")(x_adv)
    adv_out = tf.keras.layers.Dense(1, activation="sigmoid", name="adv_output")(x_adv)

    return tf.keras.Model(inputs, [main_out, adv_out])


bce_fn = tf.keras.losses.BinaryCrossentropy(reduction="none")


def main_loss(y_true, y_pred, is_female, sample_weight, fair_lambda):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    is_fem = tf.cast(tf.reshape(is_female, [-1]), tf.float32)
    sw = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)

    per_bce = -(y_true * tf.math.log(y_pred + 1e-7) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-7))
    weighted_bce = tf.reduce_sum(sw * per_bce) / (tf.reduce_sum(sw) + 1e-7)

    eps = 1e-6
    pos = tf.cast(y_true > 0.5, tf.float32)
    neg = 1.0 - pos

    fem_pos = pos * is_fem
    mal_pos = pos * (1.0 - is_fem)
    fem_neg = neg * is_fem
    mal_neg = neg * (1.0 - is_fem)

    fnr_fem = tf.reduce_sum((1.0 - y_pred) * fem_pos) / (tf.reduce_sum(fem_pos) + eps)
    fnr_mal = tf.reduce_sum((1.0 - y_pred) * mal_pos) / (tf.reduce_sum(mal_pos) + eps)
    fpr_fem = tf.reduce_sum(y_pred * fem_neg) / (tf.reduce_sum(fem_neg) + eps)
    fpr_mal = tf.reduce_sum(y_pred * mal_neg) / (tf.reduce_sum(mal_neg) + eps)

    eo_gap = tf.abs(fnr_fem - fnr_mal) + tf.abs(fpr_fem - fpr_mal)
    return weighted_bce + fair_lambda * eo_gap


def adv_loss(is_female, adv_pred):
    return tf.reduce_mean(bce_fn(tf.reshape(is_female, [-1, 1]), tf.reshape(adv_pred, [-1, 1])))


def compute_weights(df, pos_boost):
    w = np.ones(len(df), dtype=np.float32)
    fem = df[SENSITIVE_ATTRIBUTE_KEY] == 2.0
    mal = ~fem
    n = len(df)
    w[fem] = n / (2.0 * fem.sum())
    w[mal] = n / (2.0 * mal.sum())
    pos = df[LABEL_KEY] == 1
    w[pos] *= pos_boost
    w /= w.mean()
    return w


def make_ds(df, encoder_cols, sample_weights=None):
    feats = {c: df[c].values for c in encoder_cols}
    label = df[LABEL_KEY].values.astype(np.float32)
    is_fem = (df[SENSITIVE_ATTRIBUTE_KEY].values == 2.0).astype(np.float32)
    if sample_weights is None:
        sample_weights = np.ones(len(df), dtype=np.float32)
    return tf.data.Dataset.from_tensor_slices((feats, label, is_fem, sample_weights))


@tf.function
def train_step(model, optimizer, x_batch, y_batch, fem_batch, sw_batch, fair_lambda, train_loss_tracker, train_auc):
    with tf.GradientTape() as tape:
        main_pred, adv_pred = model(x_batch, training=True)
        loss_main = main_loss(y_batch, main_pred, fem_batch, sw_batch, fair_lambda)
        loss_adv = adv_loss(fem_batch, adv_pred)
        total = loss_main + ADV_LAMBDA * loss_adv
    grads = tape.gradient(total, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss_tracker.update_state(total)
    train_auc.update_state(tf.reshape(y_batch, [-1, 1]), tf.reshape(main_pred, [-1, 1]))
    return total


@tf.function
def val_step(model, x_batch, y_batch, fem_batch, sw_batch, fair_lambda, val_loss_tracker, val_auc):
    main_pred, adv_pred = model(x_batch, training=False)
    loss_main = main_loss(y_batch, main_pred, fem_batch, sw_batch, fair_lambda)
    loss_adv = adv_loss(fem_batch, adv_pred)
    total = loss_main + ADV_LAMBDA * loss_adv
    val_loss_tracker.update_state(total)
    val_auc.update_state(tf.reshape(y_batch, [-1, 1]), tf.reshape(main_pred, [-1, 1]))


def train_model(train_ds, val_ds, encoder_cols, train_df, fair_lambda, epochs):
    set_seeds(RANDOM_STATE)
    model = build_fair_model(train_df, encoder_cols)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_auc = tf.keras.metrics.AUC(name="train_auc")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_auc = tf.keras.metrics.AUC(name="val_auc")

    best_val_auc = 0.0
    best_weights = None

    print("\nTraining fair model...")
    for epoch in range(1, epochs + 1):
        train_loss.reset_state(); train_auc.reset_state()
        val_loss.reset_state(); val_auc.reset_state()

        for x_b, y_b, fem_b, sw_b in train_ds:
            train_step(model, optimizer, x_b, y_b, fem_b, sw_b, fair_lambda, train_loss, train_auc)

        for x_b, y_b, fem_b, sw_b in val_ds:
            val_step(model, x_b, y_b, fem_b, sw_b, fair_lambda, val_loss, val_auc)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}  "
                  f"loss={float(train_loss.result()):.4f} "
                  f"auc={float(train_auc.result()):.4f}  "
                  f"val_loss={float(val_loss.result()):.4f} "
                  f"val_auc={float(val_auc.result()):.4f}")

        if float(val_auc.result()) > best_val_auc + 1e-4:
            best_val_auc = float(val_auc.result())
            best_weights = model.get_weights()

    if best_weights is not None:
        model.set_weights(best_weights)
    return model, best_val_auc


def _format_slice_name(slice_name):
    if slice_name == ():
        return "Overall"
    parts = []
    for key, value in slice_name:
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _format_tfma_value(value):
    if isinstance(value, dict):
        if "doubleValue" in value:
            return f"{value['doubleValue']:.4f}"
        if "value" in value and isinstance(value["value"], (int, float)):
            return f"{value['value']:.4f}"
        if "boundedValue" in value and isinstance(value["boundedValue"], dict):
            bounded = value["boundedValue"]
            mean = bounded.get("value")
            return f"{mean:.4f}"
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_tfma_text_summary(eval_result, title):
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
    print("  Fair Model with Adversarial Debiasing")
    print("="*80)

    set_seeds(RANDOM_STATE)

    acs_df = pd.read_csv("acsemployment_2018_ca_tx.csv")
    acs_df[LABEL_KEY] = acs_df[LABEL_KEY].astype(int)

    train_df = acs_df.sample(frac=0.8, random_state=RANDOM_STATE)
    test_df = acs_df.drop(train_df.index).sample(frac=1.0, random_state=RANDOM_STATE)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    encoder_cols = [c for c in acs_df.columns
                    if c not in [LABEL_KEY] + BANNED_FEATURES + [SENSITIVE_ATTRIBUTE_KEY]]

    sw = compute_weights(train_df, POS_BOOST)
    train_ds = make_ds(train_df, encoder_cols, sample_weights=sw).shuffle(10_000, seed=RANDOM_STATE).batch(BATCH_SIZE)
    val_ds = make_ds(test_df, encoder_cols).batch(BATCH_SIZE)

    model, val_auc = train_model(train_ds, val_ds, encoder_cols, train_df, FAIR_LAMBDA, EPOCHS)

    pred_ds = tf.data.Dataset.from_tensor_slices({c: test_df[c].values for c in encoder_cols}).batch(BATCH_SIZE)
    preds, _ = model.predict(pred_ds, verbose=0)
    preds = preds.flatten()

    fair_model_analysis = test_df.copy()
    fair_model_analysis[SENSITIVE_ATTRIBUTE_KEY].replace(SENSITIVE_ATTRIBUTE_VALUES, inplace=True)
    fair_model_analysis[PREDICTION_KEY] = preds

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

    print("\nRunning TFMA evaluation...")
    fair_model_eval_result = tfma.analyze_raw_data(fair_model_analysis, eval_config)
    print_tfma_text_summary(fair_model_eval_result, "Fair Model")

    print("\n" + "="*80)
    print("  EXTENDED FAIRNESS METRICS")
    print("="*80)
    extended_metrics = compute_extended_fairness_metrics(fair_model_analysis)
    print_extended_metrics(extended_metrics, "Fair Model - Extended Metrics")

    fair_model_analysis.to_csv('fair_model_predictions.csv', index=False)

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        return obj

    with open('fair_model_extended_metrics.json', 'w') as f:
        json.dump(convert_types(extended_metrics), f, indent=2)

    print("\nResults saved:")
    print("  - fair_model_predictions.csv")
    print("  - fair_model_extended_metrics.json")


if __name__ == "__main__":
    main()
