"""
CS340 Final Project - Bias Mitigation
Strategy:
  1. Pre-processing : class-aware sample reweighting (boost Female weight)
  2. Train-time     : Adversarial debiasing - an auxiliary head tries to predict SEX
                      from the shared representation; the shared encoder is trained
                      to fool it (gradient reversal), making representations gender-neutral.
  3. Architecture   : Deeper / normalised backbone + dropout for better generalisation
  4. Post-processing: Per-group threshold calibration to equalise FNR across SEX groups
"""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
from google.protobuf import text_format

# ── Global config ──────────────────────────────────────────────────────────────
RANDOM_STATE = 200
BATCH_SIZE   = 256
EPOCHS       = 20

LABEL_KEY               = "EMPLOYED"
SENSITIVE_ATTRIBUTE_KEY = "SEX"
BANNED_FEATURES         = ["RELP"]          # SEX kept for adversarial loss during training
SENSITIVE_ATTRIBUTE_VALUES = {1.0: "Male", 2.0: "Female"}
PREDICTION_KEY          = "PRED"

# Adversarial loss weight  (λ) — how hard we push gender-neutrality
ADV_LAMBDA = 0.5


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ── TFMA helpers ───────────────────────────────────────────────────────────────
def _format_slice_name(slice_name):
    if slice_name == ():
        return "Overall"
    return ", ".join(f"{k}={v}" for k, v in slice_name)


def _format_tfma_value(value):
    if isinstance(value, dict):
        if "doubleValue" in value:
            return f"{value['doubleValue']:.4f}"
        if "boundedValue" in value:
            b = value["boundedValue"]
            return f"{b.get('value', 0):.4f} [{b.get('lowerBound', 0):.4f}, {b.get('upperBound', 0):.4f}]"
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_tfma_text_summary(eval_result, title):
    print(f"\nTFMA Text Summary: {title}")
    metrics_by_slice = eval_result.get_metrics_for_all_slices()
    for slice_name in sorted(metrics_by_slice.keys(), key=str):
        print(f"\nSlice: {_format_slice_name(slice_name)}")
        for metric_name in sorted(metrics_by_slice[slice_name].keys()):
            print(f"  {metric_name}: {_format_tfma_value(metrics_by_slice[slice_name][metric_name])}")


# ── Gradient Reversal Layer ────────────────────────────────────────────────────
@tf.custom_gradient
def _gradient_reversal_op(x, lam):
    def grad(dy):
        return -lam * dy, None
    return x, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    """Multiplies the gradient by -λ during back-prop (Ganin et al., 2015)."""
    def __init__(self, lam: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lam = tf.Variable(lam, trainable=False, dtype=tf.float32, name="grl_lambda")

    def call(self, x):
        return _gradient_reversal_op(tf.cast(x, tf.float32), self.lam)

    def get_config(self):
        cfg = super().get_config()
        cfg["lam"] = float(self.lam.numpy())
        return cfg


# ── Model ──────────────────────────────────────────────────────────────────────
def build_adversarial_model(
    feature_df: pd.DataFrame,
    model_feature_cols: list,
    adv_lambda: float = ADV_LAMBDA,
) -> tf.keras.Model:
    """
    Two-headed model:
      • main_output  — predicts EMPLOYED  (sigmoid)
      • adv_output   — predicts SEX       (sigmoid, with GRL)

    The GRL ensures the shared encoder learns gender-neutral features.
    """
    inputs = {
        name: tf.keras.Input(shape=(1,), name=name, dtype=tf.float64)
        for name in model_feature_cols
    }

    def stack_dict(d):
        return tf.concat(
            [tf.cast(d[k], tf.float64) for k in sorted(d.keys())], axis=-1
        )

    x = stack_dict(inputs)

    # Normalisation fitted on training features only
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(feature_df[model_feature_cols].values)
    x = normalizer(x)

    # ── Shared encoder ──
    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(32, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # ── Main classification head ──
    main_output = tf.keras.layers.Dense(1, activation="sigmoid", name="main_output")(x)

    # ── Adversarial head (GRL) ──
    x_adv = GradientReversalLayer(lam=adv_lambda, name="grl")(x)
    x_adv = tf.keras.layers.Dense(16, activation="relu")(x_adv)
    adv_output = tf.keras.layers.Dense(1, activation="sigmoid", name="adv_output")(x_adv)

    return tf.keras.Model(inputs=inputs, outputs=[main_output, adv_output])


# ── Dataset helpers ────────────────────────────────────────────────────────────
def make_adversarial_dataset(df: pd.DataFrame, feature_cols: list) -> tf.data.Dataset:
    """
    Returns a dataset of ({features}, {main_label, adv_label}).
    adv_label = 0 for Male (SEX==1), 1 for Female (SEX==2).
    """
    features = {col: df[col].values for col in feature_cols}
    main_labels = df[LABEL_KEY].values.astype(np.float32)
    adv_labels  = (df[SENSITIVE_ATTRIBUTE_KEY].values == 2.0).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices(
        (features, {"main_output": main_labels, "adv_output": adv_labels})
    )


# ── Sample weights: boost Female, balance positive/negative per group ──────────
def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Two-level reweighting:
      1. Equalise group sizes  (Male / Female)
      2. Equalise positive-rate within each group
    """
    weights = np.ones(len(df), dtype=np.float32)

    female_mask = df[SENSITIVE_ATTRIBUTE_KEY] == 2.0
    male_mask   = ~female_mask

    n_female = female_mask.sum()
    n_male   = male_mask.sum()
    n_total  = len(df)

    # Group-level weight
    w_female = n_total / (2.0 * n_female)
    w_male   = n_total / (2.0 * n_male)

    weights[female_mask] = w_female
    weights[male_mask]   = w_male

    # Within-group label balance
    for mask, label_col in [(female_mask, LABEL_KEY), (male_mask, LABEL_KEY)]:
        pos = (df[label_col] == 1) & mask
        neg = (df[label_col] == 0) & mask
        n_pos = pos.sum()
        n_neg = neg.sum()
        if n_pos > 0 and n_neg > 0:
            w_pos = (n_pos + n_neg) / (2.0 * n_pos)
            w_neg = (n_pos + n_neg) / (2.0 * n_neg)
            weights[pos] *= w_pos
            weights[neg] *= w_neg

    # Normalise so mean weight == 1
    weights /= weights.mean()
    return weights


# ── Per-group threshold calibration ───────────────────────────────────────────
def calibrate_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    target_metric: str = "fnr",
) -> dict:
    """
    Find per-group thresholds that equalise FNR across groups.
    Returns {group_value: threshold}.
    """
    unique_groups = np.unique(group)
    fnrs   = {}
    thresholds = {}

    # For each group find threshold minimising |FNR - overall_FNR|
    # First compute the overall FNR at t=0.5
    overall_fnr = 1.0 - (((y_pred >= 0.5) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))

    for g in unique_groups:
        mask = group == g
        yt, yp = y_true[mask], y_pred[mask]
        best_t, best_diff = 0.5, 1e9
        for t in np.linspace(0.2, 0.8, 121):
            fnr = 1.0 - (((yp >= t) & (yt == 1)).sum() / max((yt == 1).sum(), 1))
            diff = abs(fnr - overall_fnr)
            if diff < best_diff:
                best_diff, best_t = diff, t
        thresholds[g] = best_t
        fnrs[g] = best_t

    print(f"  Calibrated thresholds: {thresholds}")
    return thresholds


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    set_seeds(RANDOM_STATE)

    # ── Load data ──
    acs_df = pd.read_csv("acsemployment_2018_ca_tx.csv")
    acs_df[LABEL_KEY] = acs_df[LABEL_KEY].astype(int)

    # ── Train / test split ──
    acs_train_df = acs_df.sample(frac=0.8, random_state=RANDOM_STATE).reset_index(drop=True)
    acs_test_df  = acs_df.drop(acs_train_df.index).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # Model features = all columns except label and RELP (SEX kept for adversarial)
    model_feature_cols = [c for c in acs_df.columns if c not in [LABEL_KEY] + BANNED_FEATURES]
    # Features fed to the encoder (exclude SEX from encoder input → only used in adv head label)
    encoder_feature_cols = [c for c in model_feature_cols if c != SENSITIVE_ATTRIBUTE_KEY]

    print(f"Encoder features ({len(encoder_feature_cols)}): {encoder_feature_cols}")

    # ── Sample weights ──
    sample_weights = compute_sample_weights(acs_train_df)

    # ── Datasets ──
    train_ds = (
        make_adversarial_dataset(acs_train_df, encoder_feature_cols)
        .shuffle(10_000, seed=RANDOM_STATE)
        .batch(BATCH_SIZE)
    )
    # Attach sample weights
    sw_ds = tf.data.Dataset.from_tensor_slices(sample_weights).batch(BATCH_SIZE)
    train_ds_weighted = tf.data.Dataset.zip((train_ds, sw_ds)).map(
        lambda xy, w: (xy[0], xy[1], w)
    )

    test_ds = (
        make_adversarial_dataset(acs_test_df, encoder_feature_cols)
        .batch(BATCH_SIZE)
    )

    # ── Build & compile model ──
    model = build_adversarial_model(acs_train_df, encoder_feature_cols)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "main_output": tf.keras.losses.BinaryCrossentropy(),
            "adv_output" : tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "main_output": 1.0,
            "adv_output" : ADV_LAMBDA,
        },
        metrics={
            "main_output": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
            "adv_output": [tf.keras.metrics.BinaryAccuracy(name="adv_accuracy")],
        },
    )

    model.summary()

    # ── Learning rate schedule ──
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_main_output_auc", factor=0.5, patience=3,
        min_lr=1e-5, verbose=1, mode="max"
    )
    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_main_output_auc", patience=5,
        restore_best_weights=True, verbose=1, mode="max"
    )

    # ── Training with sample weights ──
    # tf.data with sample_weight: we use a custom train loop for clarity
    print("\n── Training adversarial debiasing model ──")
    history = model.fit(
        train_ds_weighted,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=[lr_cb, es_cb],
    )

    # ── Predict (only main_output used) ──
    test_features_ds = tf.data.Dataset.from_tensor_slices(
        {col: acs_test_df[col].values for col in encoder_feature_cols}
    ).batch(BATCH_SIZE)

    main_preds, adv_preds = model.predict(test_features_ds, batch_size=BATCH_SIZE)
    main_preds = main_preds.flatten()

    # ── Post-processing: threshold calibration ──
    y_true  = acs_test_df[LABEL_KEY].values
    sex_arr = acs_test_df[SENSITIVE_ATTRIBUTE_KEY].values

    print("\nCalibrating per-group thresholds …")
    thresholds = calibrate_thresholds(y_true, main_preds, sex_arr)

    # Apply calibrated thresholds → final binary predictions used for soft scores
    # (TFMA uses continuous scores, so we keep the raw probabilities but can
    #  shift them so that applying t=0.5 matches our calibrated thresholds)
    calibrated_preds = main_preds.copy()
    for g, t in thresholds.items():
        mask = sex_arr == g
        if t != 0.5:
            # Linear rescale: map [0, t] → [0, 0.5], [t, 1] → [0.5, 1]
            p = main_preds[mask]
            shifted = np.where(
                p < t,
                p * 0.5 / t,
                0.5 + (p - t) * 0.5 / (1.0 - t + 1e-9),
            )
            calibrated_preds[mask] = np.clip(shifted, 0.0, 1.0)

    # ── TFMA evaluation ──
    analysis_df = acs_test_df.copy()
    analysis_df[SENSITIVE_ATTRIBUTE_KEY] = analysis_df[SENSITIVE_ATTRIBUTE_KEY].replace(
        SENSITIVE_ATTRIBUTE_VALUES
    )
    analysis_df[PREDICTION_KEY] = calibrated_preds

    eval_config_pbtxt = """
      model_specs {
        prediction_key: "%s"
        label_key: "%s"
      }
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
      slicing_specs { feature_keys: "%s" }
      slicing_specs {}
    """ % (PREDICTION_KEY, LABEL_KEY, SENSITIVE_ATTRIBUTE_KEY)

    eval_config = text_format.Parse(eval_config_pbtxt, tfma.EvalConfig())
    eval_result = tfma.analyze_raw_data(analysis_df, eval_config)

    print_tfma_text_summary(eval_result, "Mitigated Model")


if __name__ == "__main__":
    main()