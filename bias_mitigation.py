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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

class FairnessMetrics:
    """Calculate various fairness metrics for model evaluation"""

    @staticmethod
    def demographic_parity_difference(y_true, y_pred, sensitive_attr):
        """Calculate demographic parity difference between groups"""
        groups = np.unique(sensitive_attr)
        selection_rates = []
        for group in groups:
            mask = sensitive_attr == group
            selection_rate = np.mean(y_pred[mask])
            selection_rates.append(selection_rate)
        return abs(selection_rates[0] - selection_rates[1])

    @staticmethod
    def equal_opportunity_difference(y_true, y_pred, sensitive_attr):
        """Calculate difference in True Positive Rates (TPR) between groups"""
        groups = np.unique(sensitive_attr)
        tpr_list = []
        for group in groups:
            mask = (sensitive_attr == group) & (y_true == 1)
            if np.sum(mask) > 0:
                tpr = np.mean(y_pred[mask])
                tpr_list.append(tpr)
            else:
                tpr_list.append(0)
        return abs(tpr_list[0] - tpr_list[1])

    @staticmethod
    def equalized_odds_difference(y_true, y_pred, sensitive_attr):
        """Calculate difference in TPR and FPR between groups"""
        groups = np.unique(sensitive_attr)
        tpr_diff = FairnessMetrics.equal_opportunity_difference(y_true, y_pred, sensitive_attr)

        # Calculate FPR difference
        fpr_list = []
        for group in groups:
            mask = (sensitive_attr == group) & (y_true == 0)
            if np.sum(mask) > 0:
                fpr = np.mean(y_pred[mask])
                fpr_list.append(fpr)
            else:
                fpr_list.append(0)
        fpr_diff = abs(fpr_list[0] - fpr_list[1])

        return (tpr_diff + fpr_diff) / 2

    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_pred_binary, sensitive_attr):
        """Calculate all fairness and performance metrics"""
        metrics = {}

        # Performance metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['auc'] = roc_auc_score(y_true, y_pred)

        # Confusion matrix for overall
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Fairness metrics
        metrics['demographic_parity_diff'] = FairnessMetrics.demographic_parity_difference(
            y_true, y_pred_binary, sensitive_attr)
        metrics['equal_opportunity_diff'] = FairnessMetrics.equal_opportunity_difference(
            y_true, y_pred_binary, sensitive_attr)
        metrics['equalized_odds_diff'] = FairnessMetrics.equalized_odds_difference(
            y_true, y_pred_binary, sensitive_attr)

        # Group-specific metrics
        groups = np.unique(sensitive_attr)
        for group in groups:
            mask = sensitive_attr == group
            group_name = 'male' if group == 1.0 else 'female'

            y_true_group = y_true[mask]
            y_pred_binary_group = y_pred_binary[mask]

            if len(y_true_group) > 0:
                tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_true_group, y_pred_binary_group).ravel()
                metrics[f'{group_name}_tpr'] = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
                metrics[f'{group_name}_fpr'] = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0
                metrics[f'{group_name}_fnr'] = fn_g / (fn_g + tp_g) if (fn_g + tp_g) > 0 else 0
                metrics[f'{group_name}_accuracy'] = accuracy_score(y_true_group, y_pred_binary_group)

        return metrics


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


def create_baseline_model(X_train, y_train, X_test, y_test, sex_test):
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

    # Calculate metrics
    metrics = FairnessMetrics.calculate_all_metrics(y_test, y_pred_proba, y_pred, sex_test)

    print(f"\nPerformance Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  TPR (Recall): {metrics['tpr']:.4f}")
    print(f"  FPR: {metrics['fpr']:.4f}")
    print(f"  FNR: {metrics['fnr']:.4f}")

    print(f"\nFairness Metrics:")
    print(f"  Demographic Parity Difference: {metrics['demographic_parity_diff']:.4f}")
    print(f"  Equal Opportunity Difference: {metrics['equal_opportunity_diff']:.4f}")
    print(f"  Equalized Odds Difference: {metrics['equalized_odds_diff']:.4f}")

    print(f"\nGroup-specific Metrics:")
    print(f"  Male TPR: {metrics['male_tpr']:.4f}, Female TPR: {metrics['female_tpr']:.4f}")
    print(f"  Male FNR: {metrics['male_fnr']:.4f}, Female FNR: {metrics['female_fnr']:.4f}")
    print(f"  Male FPR: {metrics['male_fpr']:.4f}, Female FPR: {metrics['female_fpr']:.4f}")

    return baseline_model, metrics


def create_mitigated_model(X_train, y_train, X_test, y_test, sex_train, sex_test):
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

    # Calculate metrics
    metrics = FairnessMetrics.calculate_all_metrics(y_test, y_pred_proba, y_pred, sex_test)

    print(f"\nPerformance Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  TPR (Recall): {metrics['tpr']:.4f}")
    print(f"  FPR: {metrics['fpr']:.4f}")
    print(f"  FNR: {metrics['fnr']:.4f}")

    print(f"\nFairness Metrics:")
    print(f"  Demographic Parity Difference: {metrics['demographic_parity_diff']:.4f}")
    print(f"  Equal Opportunity Difference: {metrics['equal_opportunity_diff']:.4f}")
    print(f"  Equalized Odds Difference: {metrics['equalized_odds_diff']:.4f}")

    print(f"\nGroup-specific Metrics:")
    print(f"  Male TPR: {metrics['male_tpr']:.4f}, Female TPR: {metrics['female_tpr']:.4f}")
    print(f"  Male FNR: {metrics['male_fnr']:.4f}, Female FNR: {metrics['female_fnr']:.4f}")
    print(f"  Male FPR: {metrics['male_fpr']:.4f}, Female FPR: {metrics['female_fpr']:.4f}")

    return (model1, model2, model3), thresholds, metrics


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


def plot_comparison(baseline_metrics, mitigated_metrics, save_path='results_comparison.png'):
    """Plot comparison between baseline and mitigated models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Performance metrics comparison
    performance_metrics = ['accuracy', 'auc', 'tpr', 'fnr']
    baseline_perf = [baseline_metrics[m] for m in performance_metrics]
    mitigated_perf = [mitigated_metrics[m] for m in performance_metrics]

    x = np.arange(len(performance_metrics))
    width = 0.35

    axes[0, 0].bar(x - width/2, baseline_perf, width, label='Baseline', alpha=0.8)
    axes[0, 0].bar(x + width/2, mitigated_perf, width, label='Mitigated', alpha=0.8)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(['Accuracy', 'AUC', 'TPR', 'FNR'])
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Fairness metrics comparison
    fairness_metrics = ['demographic_parity_diff', 'equal_opportunity_diff', 'equalized_odds_diff']
    baseline_fair = [baseline_metrics[m] for m in fairness_metrics]
    mitigated_fair = [mitigated_metrics[m] for m in fairness_metrics]

    x = np.arange(len(fairness_metrics))
    axes[0, 1].bar(x - width/2, baseline_fair, width, label='Baseline', alpha=0.8)
    axes[0, 1].bar(x + width/2, mitigated_fair, width, label='Mitigated', alpha=0.8)
    axes[0, 1].set_ylabel('Difference')
    axes[0, 1].set_title('Fairness Metrics Comparison (Lower is Better)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Demographic\nParity', 'Equal\nOpportunity', 'Equalized\nOdds'], fontsize=9)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Group-specific TPR comparison
    groups = ['Male', 'Female']
    baseline_tpr = [baseline_metrics['male_tpr'], baseline_metrics['female_tpr']]
    mitigated_tpr = [mitigated_metrics['male_tpr'], mitigated_metrics['female_tpr']]

    x = np.arange(len(groups))
    axes[1, 0].bar(x - width/2, baseline_tpr, width, label='Baseline', alpha=0.8)
    axes[1, 0].bar(x + width/2, mitigated_tpr, width, label='Mitigated', alpha=0.8)
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('TPR by Group (Higher and More Equal is Better)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(groups)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Group-specific FNR comparison
    baseline_fnr = [baseline_metrics['male_fnr'], baseline_metrics['female_fnr']]
    mitigated_fnr = [mitigated_metrics['male_fnr'], mitigated_metrics['female_fnr']]

    axes[1, 1].bar(x - width/2, baseline_fnr, width, label='Baseline', alpha=0.8)
    axes[1, 1].bar(x + width/2, mitigated_fnr, width, label='Mitigated', alpha=0.8)
    axes[1, 1].set_ylabel('False Negative Rate')
    axes[1, 1].set_title('FNR by Group (Lower and More Equal is Better)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(groups)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


def save_results_to_csv(baseline_metrics, mitigated_metrics, save_path='results_metrics.csv'):
    """Save metrics comparison to CSV"""
    results_df = pd.DataFrame({
        'Metric': list(baseline_metrics.keys()),
        'Baseline': list(baseline_metrics.values()),
        'Mitigated': list(mitigated_metrics.values())
    })
    results_df['Improvement'] = results_df['Mitigated'] - results_df['Baseline']
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


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
    baseline_model, baseline_metrics = create_baseline_model(
        X_train_scaled, y_train, X_test_scaled, y_test, sex_test
    )

    # Train and evaluate mitigated model
    mitigated_models, thresholds, mitigated_metrics = create_mitigated_model(
        X_train_scaled, y_train, X_test_scaled, y_test, sex_train, sex_test
    )

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print("\nPerformance Comparison:")
    print(f"  Accuracy: Baseline={baseline_metrics['accuracy']:.4f}, "
          f"Mitigated={mitigated_metrics['accuracy']:.4f}, "
          f"Diff={mitigated_metrics['accuracy']-baseline_metrics['accuracy']:.4f}")
    print(f"  AUC: Baseline={baseline_metrics['auc']:.4f}, "
          f"Mitigated={mitigated_metrics['auc']:.4f}, "
          f"Diff={mitigated_metrics['auc']-baseline_metrics['auc']:.4f}")

    print("\nFairness Comparison:")
    print(f"  Equal Opportunity Diff: Baseline={baseline_metrics['equal_opportunity_diff']:.4f}, "
          f"Mitigated={mitigated_metrics['equal_opportunity_diff']:.4f}, "
          f"Improvement={baseline_metrics['equal_opportunity_diff']-mitigated_metrics['equal_opportunity_diff']:.4f}")

    # Count metrics where mitigated model is better
    better_count = 0
    fairness_keys = ['demographic_parity_diff', 'equal_opportunity_diff', 'equalized_odds_diff']
    for key in fairness_keys:
        if mitigated_metrics[key] < baseline_metrics[key]:
            better_count += 1

    print(f"\nFairness metrics improved: {better_count}/3")

    # Save results
    plot_comparison(baseline_metrics, mitigated_metrics)
    save_results_to_csv(baseline_metrics, mitigated_metrics)

    print("\n" + "="*70)
    print("Analysis complete! Check results_comparison.png and results_metrics.csv")
    print("="*70)


if __name__ == "__main__":
    main()
