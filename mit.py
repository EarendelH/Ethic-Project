import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
from google.protobuf import text_format

RANDOM_STATE = 200
BATCH_SIZE = 100
EPOCHS = 10

LABEL_KEY = "EMPLOYED"
SENSITIVE_ATTRIBUTE_KEY = "SEX"  # ✓ 修复：补齐你代码中遗漏的全局变量定义
BANNED_FEATURES = ['SEX', 'RELP']
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

    # 1. 输入层归一化
    x = normalizer(x)
    
    # 2. 第一层：Dense + BN + ReLU
    x = tf.keras.layers.Dense(16, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    
    # 3. 第二层：Dense + BN + ReLU
    x = tf.keras.layers.Dense(32, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    
    # 4. 第三层：收尾隐藏层
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    
    # 5. 输出层：二分类概率
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


def main() -> None:
    set_seeds(RANDOM_STATE)

    acs_df = pd.read_csv("acsemployment_2018_ca_tx.csv")
    acs_df[LABEL_KEY] = acs_df[LABEL_KEY].astype(int)

    # 1. 划分训练/测试集，并显式重置索引（关键修复：杜绝不连续索引导致 Dataset 转换时的潜在错位风险）
    acs_train_df = acs_df.sample(frac=0.8, random_state=RANDOM_STATE).reset_index(drop=True)
    acs_test_df = acs_df.drop(acs_train_df.index).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # 2. 动态过滤得到没有任何 banned 特征的纯模型特征列表
    model_features_only = [col for col in acs_df.columns if col != LABEL_KEY and col not in BANNED_FEATURES]
    model_cols = model_features_only + [LABEL_KEY]

    # 3. 构造干净的训练/测试 tf.data.Dataset 管道
    train_ds = dataframe_to_dataset(acs_train_df[model_cols])
    train_batches = train_ds.batch(BATCH_SIZE)

    test_ds = dataframe_to_dataset(acs_test_df[model_cols])
    test_batches = test_ds.batch(BATCH_SIZE)

    # 4. 实例化模型（关键修复：只用训练集特征进行 normalizer.adapt，防止测试集数据泄露）
    train_features_only_df = acs_train_df[model_features_only]
    model = build_model(train_features_only_df)

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

    # 5. 预测阶段：构建只含有 15 个模型特征的测试输入流
    test_features_ds = tf.data.Dataset.from_tensor_slices(dict(acs_test_df[model_features_only]))
    test_features_batches = test_features_ds.batch(BATCH_SIZE)
    predictions = model.predict(test_features_batches, batch_size=BATCH_SIZE)

    # 6. 评估阶段：在含有原始敏感特征的测试集快照上合并预测结果
    analysis_df = acs_test_df.copy()
    
    # 修复：直接显式映射赋值，既避免了未定义报错，也安全消除了 SettingWithCopyWarning 警告风险
    analysis_df[SENSITIVE_ATTRIBUTE_KEY] = analysis_df[SENSITIVE_ATTRIBUTE_KEY].replace(SENSITIVE_ATTRIBUTE_VALUES)
    
    # 修复：如果是老版本 Keras，predictions 返回的二维形状 (N, 1) 在 pandas 赋值时会报维度警报或错位，flatten() 确保安全
    analysis_df[PREDICTION_KEY] = predictions.flatten()

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