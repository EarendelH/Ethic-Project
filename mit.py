"""
CS340 Final Project - Bias Mitigation v3

Strategy:
  1. Pre-processing : group + label reweighting
  2. Train-time A   : adversarial debiasing (GRL)
  3. Train-time B   : FNR-gap penalty in main loss
  4. Custom GradientTape training loop — avoids all Keras sample_weight shape
     restrictions; gives full control over what each head receives.
  5. No post-processing threshold shifts
"""

import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
from google.protobuf import text_format

RANDOM_STATE = 200
BATCH_SIZE   = 256
EPOCHS       = 25

LABEL_KEY               = "EMPLOYED"
SENSITIVE_ATTRIBUTE_KEY = "SEX"
BANNED_FEATURES         = ["RELP"]
SENSITIVE_ATTRIBUTE_VALUES = {1.0: "Male", 2.0: "Female"}
PREDICTION_KEY          = "PRED"

ADV_LAMBDA  = 0.6   # adversarial head loss weight
FAIR_LAMBDA = 1.5   # FNR-gap penalty weight


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _fmt_slice(s):
    return "Overall" if s == () else ", ".join(f"{k}={v}" for k, v in s)

def _fmt_val(v):
    if isinstance(v, dict):
        if "doubleValue" in v: return f"{v['doubleValue']:.4f}"
        if "boundedValue" in v: return f"{v['boundedValue'].get('value', 0):.4f}"
    return f"{v:.4f}" if isinstance(v, float) else str(v)

def print_tfma(result, title):
    print(f"\nTFMA Text Summary: {title}")
    for sname in sorted(result.get_metrics_for_all_slices().keys(), key=str):
        print(f"\nSlice: {_fmt_slice(sname)}")
        m = result.get_metrics_for_all_slices()[sname]
        for mn in sorted(m.keys()):
            print(f"  {mn}: {_fmt_val(m[mn])}")


# ── Gradient Reversal Layer ────────────────────────────────────────────────────
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


# ── Model (build only, no compile) ────────────────────────────────────────────
def build_model(train_df, encoder_cols):
    inputs = {c: tf.keras.Input(shape=(1,), name=c, dtype=tf.float64)
              for c in encoder_cols}

    def stack(d):
        return tf.concat(
            [tf.cast(d[k], tf.float64) for k in sorted(d.keys())], axis=-1
        )

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


# ── Loss functions (plain functions, not Keras Loss objects) ───────────────────
bce_fn = tf.keras.losses.BinaryCrossentropy(reduction="none")

def main_loss(y_true, y_pred, is_female, sample_weight):
    """BCE weighted by sample_weight + FAIR_LAMBDA * soft-FNR gap."""
    y_true   = tf.cast(tf.reshape(y_true,   [-1]), tf.float32)
    y_pred   = tf.cast(tf.reshape(y_pred,   [-1]), tf.float32)
    is_fem   = tf.cast(tf.reshape(is_female, [-1]), tf.float32)
    sw       = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)

    # Weighted BCE
    per_sample_bce = -(
        y_true       * tf.math.log(y_pred + 1e-7) +
        (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
    )
    weighted_bce = tf.reduce_sum(sw * per_sample_bce) / (tf.reduce_sum(sw) + 1e-7)

    # Soft FNR per group: mean(1-p | y=1, group=g)
    pos     = tf.cast(y_true > 0.5, tf.float32)
    eps     = 1e-6
    fem_pos = pos * is_fem
    mal_pos = pos * (1.0 - is_fem)

    fnr_fem = tf.reduce_sum((1.0 - y_pred) * fem_pos) / (tf.reduce_sum(fem_pos) + eps)
    fnr_mal = tf.reduce_sum((1.0 - y_pred) * mal_pos) / (tf.reduce_sum(mal_pos) + eps)

    gap = tf.abs(fnr_fem - fnr_mal)

    return weighted_bce + FAIR_LAMBDA * gap

def adv_loss(is_female, adv_pred):
    """Standard BCE for the adversarial head."""
    return tf.reduce_mean(
        bce_fn(tf.reshape(is_female, [-1, 1]),
               tf.reshape(adv_pred, [-1, 1]))
    )


# ── Dataset ────────────────────────────────────────────────────────────────────
def compute_weights(df):
    w = np.ones(len(df), dtype=np.float32)
    fem = df[SENSITIVE_ATTRIBUTE_KEY] == 2.0
    mal = ~fem
    n   = len(df)
    w[fem] = n / (2.0 * fem.sum())
    w[mal] = n / (2.0 * mal.sum())
    for mask in [fem, mal]:
        pos = (df[LABEL_KEY] == 1) & mask
        neg = (df[LABEL_KEY] == 0) & mask
        np_, nn = pos.sum(), neg.sum()
        if np_ > 0 and nn > 0:
            w[pos] *= (np_ + nn) / (2.0 * np_)
            w[neg] *= (np_ + nn) / (2.0 * nn)
    w /= w.mean()
    return w

def make_ds(df, encoder_cols, sample_weights=None):
    """Each element: (feature_dict, label, is_female, sample_weight)"""
    feats  = {c: df[c].values for c in encoder_cols}
    label  = df[LABEL_KEY].values.astype(np.float32)
    is_fem = (df[SENSITIVE_ATTRIBUTE_KEY].values == 2.0).astype(np.float32)
    if sample_weights is None:
        sample_weights = np.ones(len(df), dtype=np.float32)
    return tf.data.Dataset.from_tensor_slices((feats, label, is_fem, sample_weights))


# ── Custom training loop ───────────────────────────────────────────────────────
@tf.function
def train_step(model, optimizer, x_batch, y_batch, fem_batch, sw_batch,
               train_loss_tracker, train_auc):
    with tf.GradientTape() as tape:
        main_pred, adv_pred = model(x_batch, training=True)
        loss_main = main_loss(y_batch, main_pred, fem_batch, sw_batch)
        loss_adv  = adv_loss(fem_batch, adv_pred)
        total     = loss_main + ADV_LAMBDA * loss_adv
    grads = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss_tracker.update_state(total)
    train_auc.update_state(tf.reshape(y_batch, [-1, 1]),
                           tf.reshape(main_pred, [-1, 1]))
    return total

@tf.function
def val_step(model, x_batch, y_batch, fem_batch, sw_batch,
             val_loss_tracker, val_auc):
    main_pred, adv_pred = model(x_batch, training=False)
    loss_main = main_loss(y_batch, main_pred, fem_batch, sw_batch)
    loss_adv  = adv_loss(fem_batch, adv_pred)
    total     = loss_main + ADV_LAMBDA * loss_adv
    val_loss_tracker.update_state(total)
    val_auc.update_state(tf.reshape(y_batch, [-1, 1]),
                         tf.reshape(main_pred, [-1, 1]))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    set_seeds(RANDOM_STATE)

    acs_df = pd.read_csv("acsemployment_2018_ca_tx.csv")
    acs_df[LABEL_KEY] = acs_df[LABEL_KEY].astype(int)

    train_df = acs_df.sample(frac=0.8, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df  = acs_df.drop(train_df.index).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    encoder_cols = [c for c in acs_df.columns
                    if c not in [LABEL_KEY] + BANNED_FEATURES + [SENSITIVE_ATTRIBUTE_KEY]]
    print(f"Encoder features ({len(encoder_cols)}): {encoder_cols}")

    sw = compute_weights(train_df)

    train_ds = (
        make_ds(train_df, encoder_cols, sample_weights=sw)
        .shuffle(10_000, seed=RANDOM_STATE)
        .batch(BATCH_SIZE)
    )
    val_ds = make_ds(test_df, encoder_cols).batch(BATCH_SIZE)

    model     = build_model(train_df, encoder_cols)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    # Metrics
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_auc  = tf.keras.metrics.AUC(name="train_auc")
    val_loss   = tf.keras.metrics.Mean(name="val_loss")
    val_auc    = tf.keras.metrics.AUC(name="val_auc")

    best_val_auc   = 0.0
    patience_count = 0
    PATIENCE       = 7
    best_weights   = None

    print("\n── Training v3 ──")
    for epoch in range(1, EPOCHS + 1):
        train_loss.reset_state(); train_auc.reset_state()
        val_loss.reset_state();   val_auc.reset_state()

        for x_b, y_b, fem_b, sw_b in train_ds:
            train_step(model, optimizer, x_b, y_b, fem_b, sw_b,
                       train_loss, train_auc)

        for x_b, y_b, fem_b, sw_b in val_ds:
            val_step(model, x_b, y_b, fem_b, sw_b, val_loss, val_auc)

        tl = float(train_loss.result()); ta = float(train_auc.result())
        vl = float(val_loss.result());   va = float(val_auc.result())
        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"loss={tl:.4f} auc={ta:.4f}  "
              f"val_loss={vl:.4f} val_auc={va:.4f}")

        # LR reduction
        if epoch > 1 and epoch % 4 == 0:
            old_lr = float(optimizer.learning_rate)
            if va <= best_val_auc:
                new_lr = max(old_lr * 0.5, 1e-6)
                optimizer.learning_rate.assign(new_lr)
                if new_lr != old_lr:
                    print(f"  LR {old_lr:.2e} → {new_lr:.2e}")

        # Early stopping
        if va > best_val_auc + 1e-4:
            best_val_auc   = va
            patience_count = 0
            best_weights   = model.get_weights()
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val_auc={best_val_auc:.4f})")
                break

    if best_weights is not None:
        model.set_weights(best_weights)
        print(f"Restored best weights (val_auc={best_val_auc:.4f})")

    # ── Predict ──
    pred_ds = tf.data.Dataset.from_tensor_slices(
        {c: test_df[c].values for c in encoder_cols}
    ).batch(BATCH_SIZE)
    preds, _ = model.predict(pred_ds)
    preds = preds.flatten()

    # ── TFMA ──
    analysis_df = test_df.copy()
    analysis_df[SENSITIVE_ATTRIBUTE_KEY] = (
        analysis_df[SENSITIVE_ATTRIBUTE_KEY].replace(SENSITIVE_ATTRIBUTE_VALUES)
    )
    analysis_df[PREDICTION_KEY] = preds

    cfg_pbtxt = """
      model_specs { prediction_key: "%s" label_key: "%s" }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        metrics { class_name: "BinaryAccuracy" }
        metrics { class_name: "AUC" }
        metrics { class_name: "ConfusionMatrixPlot" }
        metrics { class_name: "FairnessIndicators"
                  config: '{"thresholds": [0.50]}' }
      }
      slicing_specs { feature_keys: "%s" }
      slicing_specs {}
    """ % (PREDICTION_KEY, LABEL_KEY, SENSITIVE_ATTRIBUTE_KEY)

    result = tfma.analyze_raw_data(
        analysis_df, text_format.Parse(cfg_pbtxt, tfma.EvalConfig())
    )
    print_tfma(result, "Mitigated Model v3")


if __name__ == "__main__":
    main()