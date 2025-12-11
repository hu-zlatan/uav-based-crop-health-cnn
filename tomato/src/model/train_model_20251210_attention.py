# ==============================================================================
# Tomato Leaf Disease Recognition Model with Attention Mechanism (v2.0)
# File: train_model_20251210_attention.py
# Path: src/model/train_model_20251210_attention.py
# Features: 
# 1. Integrated CBAM Attention for Image Feature Extraction
# 2. Enhanced Data Augmentation for Plant Leaf Images
# 3. AdamW Optimizer with Explicit Weight Decay
# 4. Progressive Fine-tuning Strategy
# 5. Comprehensive Metrics (Accuracy/Precision/Recall)
# ==============================================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# ===================== 核心路径配置 =====================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

# ===================== 训练参数配置（优化后）=====================
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
BASE_EPOCHS = 60  # 基础训练轮次
FINE_TUNE_EPOCHS = 100  # 微调轮次
LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5  # 微调初始学习率
NUM_CLASSES = None
LABEL_TO_IDX = None

# ===================== 图像注意力机制：CBAM简化版（适配CNN）=====================
class ChannelAttention(layers.Layer):
    """通道注意力机制（SE Attention），适配图像特征"""
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.fc1 = layers.Dense(self.channels // self.ratio, activation="relu")
        self.fc2 = layers.Dense(self.channels, activation="sigmoid")
        super().build(input_shape)

    def call(self, x):
        # 全局平均池化 + 全局最大池化
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        
        # 全连接层提取通道注意力权重
        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))
        
        # 权重融合 + 特征加权
        attention = layers.Add()([avg_out, max_out])
        attention = layers.Reshape((1, 1, self.channels))(attention)
        return x * attention

    def compute_output_shape(self, input_shape):
        return input_shape

class SpatialAttention(layers.Layer):
    """空间注意力机制，适配图像特征"""
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding="same", activation="sigmoid")
        super().build(input_shape)

    def call(self, x):
        # 通道维度的平均池化 + 最大池化
        avg_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(x)
        max_pool = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(x)
        
        # 卷积提取空间注意力权重
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv(concat)
        
        # 特征加权
        return x * attention

    def compute_output_shape(self, input_shape):
        return input_shape

class CBAMAttention(layers.Layer):
    """CBAM注意力（通道+空间），适配CNN图像特征提取"""
    def __init__(self, ratio=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channel_att = ChannelAttention(ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# ===================== 工具函数：加载并预处理数据集 =====================
def load_and_preprocess_data():
    global NUM_CLASSES, LABEL_TO_IDX
    
    label_to_idx_path = os.path.join(PROCESSED_DIR, "label_to_idx.npy")
    if not os.path.exists(label_to_idx_path):
        raise FileNotFoundError(
            f"❌ 标签映射文件不存在：{label_to_idx_path}\n"
            "请先执行：python src/data/prepare_tomato_data.py"
        )
    LABEL_TO_IDX = np.load(label_to_idx_path, allow_pickle=True).item()
    NUM_CLASSES = len(LABEL_TO_IDX)
    print(f"✅ 加载标签映射完成 | 类别数：{NUM_CLASSES} | 映射关系：{LABEL_TO_IDX}")
    
    train_csv_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_csv_path = os.path.join(PROCESSED_DIR, "val.csv")
    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        raise FileNotFoundError(
            f"❌ 训练/验证集CSV不存在\n"
            f"缺失文件：{train_csv_path if not os.path.exists(train_csv_path) else val_csv_path}\n"
            "请先执行数据预处理脚本"
        )
    
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    print(f"✅ 加载数据集完成 | 训练集：{len(train_df)} 张 | 验证集：{len(val_df)} 张")
    
    def process_dataframe(df, desc):
        images = []
        labels = []
        for idx, row in tqdm(df.iterrows(), desc=desc, total=len(df)):
            img_rel_path = row["img_path"]
            img_abs_path = os.path.join(ROOT_DIR, img_rel_path)
            
            if not os.path.exists(img_abs_path):
                print(f"⚠️  图片不存在，跳过：{img_abs_path}")
                continue
            
            img = load_img(img_abs_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0
            
            images.append(img_array)
            labels.append(row["label"])
        
        images = np.array(images, dtype=np.float32)
        labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
        
        return images, labels
    
    X_train, y_train = process_dataframe(train_df, "加载训练集图片")
    X_val, y_val = process_dataframe(val_df, "加载验证集图片")
    
    print(f"✅ 图片预处理完成：")
    print(f"   - 训练集：{X_train.shape} | 标签：{y_train.shape}")
    print(f"   - 验证集：{X_val.shape} | 标签：{y_val.shape}")
    
    return X_train, y_train, X_val, y_val

# ===================== 工具函数：构建迁移学习模型（修正注意力机制）=====================
def build_tomato_model():
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        weights="imagenet",
        include_top=False
    )
    base_model.trainable = False
    
    # 增强数据增强策略
    model = models.Sequential([
        # 数据增强层
        layers.RandomFlip("horizontal_and_vertical", input_shape=(*IMAGE_SIZE, 3)),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2, 0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1),
        
        # 基础特征提取
        base_model,
        
        # 核心改进：CBAM注意力机制（适配图像CNN）
        CBAMAttention(ratio=16, kernel_size=7),
        
        # 特征聚合与分类头
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    
    # 使用AdamW优化器增强正则化效果
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), 
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    print("\n📌 模型结构概览（v2.0 | 含CBAM注意力机制）：")
    model.summary(expand_nested=True)
    
    return model, base_model

# ===================== 工具函数：执行模型训练 =====================
def train_tomato_model():
    X_train, y_train, X_val, y_val = load_and_preprocess_data()
    model, base_model = build_tomato_model()
    
    # 优化回调函数（保留原文件名）
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(PROCESSED_DIR, "best_tomato_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(ROOT_DIR, "logs", "tomato_model"),
            histogram_freq=1
        )
    ]
    
    # 基础训练
    print("\n🚀 开始基础训练（v2.0 | 冻结预训练层）...")
    history_base = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=BASE_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 模型微调
    print("\n🔧 开始模型微调（v2.0 | 解冻更多层）...")
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # 微调阶段优化器（带学习率衰减）
    fine_tune_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=FINE_TUNE_LR,
            decay_steps=10000,
            decay_rate=0.9
        ),
        weight_decay=1e-4
    )
    
    model.compile(
        optimizer=fine_tune_optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), 
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    # 继续微调训练
    history_fine = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=BASE_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history_base.epoch[-1],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 合并训练历史（保留原文件名）
    all_history = {
        "loss": history_base.history["loss"] + history_fine.history["loss"],
        "val_loss": history_base.history["val_loss"] + history_fine.history["val_loss"],
        "accuracy": history_base.history["accuracy"] + history_fine.history["accuracy"],
        "val_accuracy": history_base.history["val_accuracy"] + history_fine.history["val_accuracy"],
        "precision": history_base.history["precision"] + history_fine.history["precision"],
        "recall": history_base.history["recall"] + history_fine.history["recall"]
    }
    history_df = pd.DataFrame(all_history)
    history_df.to_csv(os.path.join(PROCESSED_DIR, "train_history.csv"), index=False)
    
    # 最终评估
    print("\n📊 训练完成 | v2.0模型最终验证集评估：")
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
    print(f"   - 验证集精度：{val_acc:.4f}")
    print(f"   - 验证集损失：{val_loss:.4f}")
    print(f"   - 验证集精确率：{val_precision:.4f}")
    print(f"   - 验证集召回率：{val_recall:.4f}")
    
    print("\n📁 v2.0版本生成文件清单（保持原命名）：")
    print(f"   ✅ 最佳模型：{os.path.join(PROCESSED_DIR, 'best_tomato_model.keras')}")
    print(f"   ✅ 训练历史：{os.path.join(PROCESSED_DIR, 'train_history.csv')}")
    print(f"   ✅ TensorBoard日志：{os.path.join(ROOT_DIR, 'logs', 'tomato_model')}")
    
    return model, history_base, history_fine

# ===================== 主函数：启动训练 =====================
if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(
            f"❌ 预处理目录不存在：{PROCESSED_DIR}\n"
            "请先执行数据预处理脚本：python src/data/prepare_tomato_data.py"
        )
    
    try:
        model, history_base, history_fine = train_tomato_model()
        print("\n🎉 番茄胁迫识别模型（v2.0 | 20251210 | CBAM注意力机制版）训练全部完成！")
    except Exception as e:
        print(f"\n❌ v2.0模型训练过程出错：{str(e)}")
        raise