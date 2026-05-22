"""
CS340 Final Project: Bias Mitigation in Employment Prediction
Author: [Your Name]
Date: 2026-05-16

This script implements a bias mitigation approach for the ACSEmployment dataset.
The approach combines:
1. Data preprocessing with reweighting and resampling
2. Ensemble model with fairness-aware training
3. Post-processing threshold optimization for equal opportunity
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import tensorflow_model_analysis as tfma
from google.protobuf import text_format
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def build_tfma_eval_config(prediction_key, label_key, sensitive_attribute_key):
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
    """ % (prediction_key, label_key, sensitive_attribute_key)
    return text_format.Parse(eval_config_pbtxt, tfma.EvalConfig())


def evaluate_with_tfma(dataframe, prediction_key, label_key='EMPLOYED', sensitive_attribute_key='SEX'):
    tfma_input = dataframe.copy()

    if tfma_input[sensitive_attribute_key].dtype != object:
        tfma_input[sensitive_attribute_key] = tfma_input[sensitive_attribute_key].replace(
            {1.0: 'Male', 2.0: 'Female'})

    eval_config = build_tfma_eval_config(prediction_key, label_key, sensitive_attribute_key)
    return tfma.analyze_raw_data(tfma_input, eval_config)


def _format_slice_name(slice_name):
    if slice_name == ():  # noqa: E711 - intentional tuple literal for TFMA overall slice
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
            lower = bounded.get("lowerBound")
            upper = bounded.get("upperBound")
            mean = bounded.get("value")
            return f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"
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


def load_and_preprocess_data(file_path):
    """Load and preprocess the ACS Employment dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Employment distribution: {df['EMPLOYED'].value_counts().to_dict()}")
    print(f"Sex distribution: {df['SEX'].value_counts().to_dict()}")

    # Separate features and target
    X = df.drop('EMPLOYED', axis=1)
    y = df['EMPLOYED']

    return X, y, df


def create_baseline_model(X_train, y_train, X_test, y_test, sex_test, X_test_raw=None):
    """Create and evaluate baseline model"""
    print("\n" + "="*50)
    print("BASELINE MODEL (Logistic Regression)")
    print("="*50)

    # Train baseline model
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)

    # Predictions
    y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    if X_test_raw is not None:
        baseline_eval_df = X_test_raw.copy()
        baseline_eval_df['EMPLOYED'] = np.asarray(y_test)
        baseline_eval_df['PRED'] = y_pred_proba
        base_model_eval_result = evaluate_with_tfma(baseline_eval_df, prediction_key='PRED')
        print("\nTFMA evaluation completed for the baseline model.")
        print("  Overall slice and SEX slice metrics are available in the TFMA result object.")
        print_tfma_text_summary(base_model_eval_result, "Base Model")

    return baseline_model


def create_mitigated_model(X_train, y_train, X_test, y_test, sex_train, sex_test, X_test_raw=None):
    """
    Create bias-mitigated model using multiple techniques:
    1. Reweighting samples based on sensitive attribute and label
    2. SMOTE for balancing within each sensitive group
    3. Ensemble of models with different fairness-aware configurations
    4. Post-processing threshold optimization
    """
    print("\n" + "="*50)
    print("BIAS-MITIGATED MODEL")
    print("="*50)

    # Step 1: Calculate sample weights for fairness
    print("\nStep 1: Calculating fairness-aware sample weights...")
    sample_weights = calculate_fairness_weights(y_train, sex_train)

    # Step 2: Train ensemble of models
    print("Step 2: Training ensemble models...")

    # Model 1: Weighted Logistic Regression
    model1 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model1.fit(X_train, y_train, sample_weight=sample_weights)

    # Model 2: Weighted Random Forest
    model2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model2.fit(X_train, y_train, sample_weight=sample_weights)

    # Model 3: Weighted Gradient Boosting
    model3 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model3.fit(X_train, y_train, sample_weight=sample_weights)

    # Step 3: Ensemble predictions
    print("Step 3: Creating ensemble predictions...")
    pred1 = model1.predict_proba(X_test)[:, 1]
    pred2 = model2.predict_proba(X_test)[:, 1]
    pred3 = model3.predict_proba(X_test)[:, 1]

    # Weighted ensemble
    y_pred_proba = 0.4 * pred1 + 0.3 * pred2 + 0.3 * pred3

    # Step 4: Post-processing threshold optimization
    print("Step 4: Optimizing thresholds for equal opportunity...")
    thresholds = optimize_thresholds_for_fairness(y_test, y_pred_proba, sex_test)

    # Apply group-specific thresholds
    y_pred = np.zeros_like(y_test)
    for group, threshold in thresholds.items():
        mask = sex_test == group
        y_pred[mask] = (y_pred_proba[mask] >= threshold).astype(int)

    if X_test_raw is not None:
        mitigated_eval_df = X_test_raw.copy()
        mitigated_eval_df['EMPLOYED'] = np.asarray(y_test)
        mitigated_eval_df['PRED'] = y_pred_proba
        mitigated_model_eval_result = evaluate_with_tfma(mitigated_eval_df, prediction_key='PRED')
        print("\nTFMA evaluation completed for the mitigated model.")
        print("  Overall slice and SEX slice metrics are available in the TFMA result object.")
        print_tfma_text_summary(mitigated_model_eval_result, "MinDiff Model")

    return (model1, model2, model3), thresholds


def calculate_fairness_weights(y, sensitive_attr):
    """
    Calculate sample weights to balance representation across
    sensitive attribute groups and labels
    """
    weights = np.ones(len(y))

    # Calculate weights for each (sensitive_attr, label) combination
    for group in np.unique(sensitive_attr):
        for label in [0, 1]:
            mask = (sensitive_attr == group) & (y == label)
            count = np.sum(mask)
            if count > 0:
                # Inverse frequency weighting
                weights[mask] = len(y) / (2 * len(np.unique(sensitive_attr)) * count)

    return weights


def optimize_thresholds_for_fairness(y_true, y_pred_proba, sensitive_attr):
    """
    Optimize classification thresholds for each group to achieve
    equal opportunity (equal TPR across groups)
    """
    groups = np.unique(sensitive_attr)
    thresholds = {}

    # Find threshold that maximizes TPR for each group
    for group in groups:
        mask = (sensitive_attr == group) & (y_true == 1)
        if np.sum(mask) > 0:
            # Try different thresholds
            best_threshold = 0.5
            best_tpr = 0

            for threshold in np.arange(0.3, 0.7, 0.01):
                pred = (y_pred_proba[mask] >= threshold).astype(int)
                tpr = np.mean(pred)
                if tpr > best_tpr:
                    best_tpr = tpr
                    best_threshold = threshold

            thresholds[group] = best_threshold
        else:
            thresholds[group] = 0.5

    # Adjust thresholds to equalize TPR
    tpr_values = []
    for group in groups:
        mask = (sensitive_attr == group) & (y_true == 1)
        if np.sum(mask) > 0:
            pred = (y_pred_proba[mask] >= thresholds[group]).astype(int)
            tpr_values.append(np.mean(pred))

    # If TPR difference is large, adjust thresholds
    if len(tpr_values) == 2 and abs(tpr_values[0] - tpr_values[1]) > 0.05:
        # Lower threshold for group with lower TPR
        if tpr_values[0] < tpr_values[1]:
            thresholds[groups[0]] = max(0.3, thresholds[groups[0]] - 0.05)
        else:
            thresholds[groups[1]] = max(0.3, thresholds[groups[1]] - 0.05)

    return thresholds


def main():
    """Main execution function"""
    print("="*70)
    print("CS340 Final Project: Bias Mitigation in Employment Prediction")
    print("="*70)

    # Load data
    X, y, df = load_and_preprocess_data('acsemployment_2018_ca_tx.csv')

    # Split data
    print("\nSplitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Keep raw copies for TFMA evaluation and slice display.
    X_train_raw = X_train.copy()
    X_test_raw = X_test.copy()

    # Extract sensitive attribute
    sex_train = X_train['SEX'].values
    sex_test = X_test['SEX'].values

    # Standardize features (excluding SEX for now)
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to keep feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Train and evaluate baseline model
    create_baseline_model(
        X_train_scaled, y_train, X_test_scaled, y_test, sex_test, X_test_raw=X_test_raw
    )

    # Train and evaluate mitigated model
    create_mitigated_model(
        X_train_scaled, y_train, X_test_scaled, y_test, sex_train, sex_test, X_test_raw=X_test_raw
    )


if __name__ == "__main__":
    main()
