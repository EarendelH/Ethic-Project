import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
from google.protobuf import text_format


RANDOM_STATE = 200
BATCH_SIZE = 100
EPOCHS = 12

LABEL_KEY = "EMPLOYED"
SENSITIVE_ATTRIBUTE_KEY = "SEX"
SENSITIVE_ATTRIBUTE_VALUES = {1.0: "Male", 2.0: "Female"}
PREDICTION_KEY = "PRED"

def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Determinism might not be available in all TensorFlow builds.
        pass


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


def dataframe_to_dataset(dataframe: pd.DataFrame) -> tf.data.Dataset:
    dataframe = dataframe.copy()
    labels = dataframe.pop(LABEL_KEY)
    return tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))


def build_model(features: pd.DataFrame) -> tf.keras.Model:
    inputs = {}
    for name in features.columns:
        if name != LABEL_KEY:
            inputs[name] = tf.keras.Input(
                shape=(1,), name=name, dtype=tf.float64
            )

    def stack_dict(inputs_dict, fun=tf.stack):
        values = []
        for key in sorted(inputs_dict.keys()):
            values.append(tf.cast(inputs_dict[key], tf.float64))
        return fun(values, axis=-1)

    x = stack_dict(inputs, fun=tf.concat)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(stack_dict(dict(features)))

    x = normalizer(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


def main() -> None:
    set_seeds(RANDOM_STATE)

    acs_df = pd.read_csv("acsemployment_2018_ca_tx.csv")
    acs_df[LABEL_KEY] = acs_df[LABEL_KEY].astype(int)

    features = acs_df.copy()
    features.pop(LABEL_KEY)

    acs_train_df = acs_df.sample(frac=0.8, random_state=RANDOM_STATE)
    acs_test_df = acs_df.drop(acs_train_df.index).sample(frac=1.0)

    train_ds = dataframe_to_dataset(acs_train_df)
    train_batches = train_ds.batch(BATCH_SIZE)

    test_ds = dataframe_to_dataset(acs_test_df)
    test_batches = test_ds.batch(BATCH_SIZE)

    model = build_model(features)

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    model.fit(train_batches, epochs=EPOCHS)
    model.evaluate(test_batches, batch_size=BATCH_SIZE)

    predictions = model.predict(test_batches, batch_size=BATCH_SIZE)

    analysis_df = acs_test_df.copy()
    analysis_df[SENSITIVE_ATTRIBUTE_KEY].replace(
        SENSITIVE_ATTRIBUTE_VALUES, inplace=True
    )
    analysis_df[PREDICTION_KEY] = predictions

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
    eval_result = tfma.analyze_raw_data(analysis_df, eval_config)

    print_tfma_text_summary(eval_result, "Mitigated Model")


if __name__ == "__main__":
    main()
