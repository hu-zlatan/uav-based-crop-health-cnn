import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# ===================== 核心路径配置（适配src/model/train_model.py） =====================
# 脚本当前路径：src/model/train_model.py
# 项目根目录：向上三级 → D:\Project-temp\uav-based-crop-health-cnn\tomato\
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# 预处理输出目录：tomato/data/processed
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
# 原始数据目录：tomato/data/raw
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

# ===================== 训练参数配置 =====================
IMAGE_SIZE = (256, 256)  # 与预处理一致的图片尺寸
BATCH_SIZE = 32           # 根据GPU显存调整（显存不足设为16/8）
EPOCHS = 20               # 基础训练轮数
LEARNING_RATE = 1e-4      # 基础学习率
NUM_CLASSES = None        # 自动识别类别数
LABEL_TO_IDX = None       # 标签编码映射字典

# ===================== 工具函数：加载并预处理数据集 =====================
def load_and_preprocess_data():
    """
    加载train.csv/val.csv，生成模型可训练的图片数组+One-Hot标签
    返回：X_train, y_train, X_val, y_val
    """
    global NUM_CLASSES, LABEL_TO_IDX
    
    # 1. 加载标签编码映射（从预处理目录）
    label_to_idx_path = os.path.join(PROCESSED_DIR, "label_to_idx.npy")
    if not os.path.exists(label_to_idx_path):
        raise FileNotFoundError(
            f"❌ 标签映射文件不存在：{label_to_idx_path}\n"
            "请先执行：python src/data/prepare_tomato_data.py"
        )
    LABEL_TO_IDX = np.load(label_to_idx_path, allow_pickle=True).item()
    NUM_CLASSES = len(LABEL_TO_IDX)
    print(f"✅ 加载标签映射完成 | 类别数：{NUM_CLASSES} | 映射关系：{LABEL_TO_IDX}")
    
    # 2. 加载训练/验证集CSV文件
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
    
    # 3. 加载并预处理图片（统一路径拼接逻辑）
    def process_dataframe(df, desc):
        """处理单个DataFrame，返回图片数组和One-Hot标签"""
        images = []
        labels = []
        for idx, row in tqdm(df.iterrows(), desc=desc, total=len(df)):
            # 拼接图片绝对路径：项目根目录 + 相对路径（如data/raw/Tomato___Healthy/xxx.jpg）
            img_rel_path = row["img_path"]
            img_abs_path = os.path.join(ROOT_DIR, img_rel_path)
            
            # 校验图片是否存在
            if not os.path.exists(img_abs_path):
                print(f"⚠️  图片不存在，跳过：{img_abs_path}")
                continue
            
            # 加载图片并预处理
            img = load_img(img_abs_path, target_size=IMAGE_SIZE)  # 调整尺寸
            img_array = img_to_array(img) / 255.0  # 归一化到0-1
            
            images.append(img_array)
            labels.append(row["label"])
        
        # 转换为numpy数组 + One-Hot编码标签
        images = np.array(images, dtype=np.float32)
        labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
        
        return images, labels
    
    # 处理训练集和验证集
    X_train, y_train = process_dataframe(train_df, "加载训练集图片")
    X_val, y_val = process_dataframe(val_df, "加载验证集图片")
    
    # 输出数据维度信息
    print(f"✅ 图片预处理完成：")
    print(f"   - 训练集：{X_train.shape} | 标签：{y_train.shape}")
    print(f"   - 验证集：{X_val.shape} | 标签：{y_val.shape}")
    
    return X_train, y_train, X_val, y_val

# ===================== 工具函数：构建迁移学习模型 =====================
def build_tomato_model():
    """构建基于MobileNetV2的迁移学习模型，适配256×256尺寸"""
    # 1. 加载预训练骨干网络（冻结底层权重）
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        weights="imagenet",  # 使用ImageNet预训练权重
        include_top=False     # 不包含顶层分类器
    )
    base_model.trainable = False  # 先冻结，训练后期微调
    
    # 2. 构建完整模型（数据增强 + 特征提取 + 分类）
    model = models.Sequential([
        # 数据增强层（仅训练阶段生效）
        layers.RandomFlip("horizontal", input_shape=(*IMAGE_SIZE, 3)),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
        
        # 预训练骨干
        base_model,
        
        # 特征聚合与分类
        layers.GlobalAveragePooling2D(),  # 全局平均池化，降低参数量
        layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.5),  # Dropout防止过拟合
        layers.Dense(NUM_CLASSES, activation="softmax")  # 分类输出层
    ])
    
    # 3. 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # 打印模型结构
    print("\n📌 模型结构概览：")
    model.summary(expand_nested=True)
    
    return model, base_model

# ===================== 工具函数：执行模型训练 =====================
def train_tomato_model():
    """主训练流程：加载数据 → 构建模型 → 训练 → 微调 → 保存"""
    # 1. 加载预处理数据
    X_train, y_train, X_val, y_val = load_and_preprocess_data()
    
    # 2. 构建模型
    model, base_model = build_tomato_model()
    
    # 3. 定义训练回调函数
    callbacks = [
        # 早停：验证集精度5轮不提升则停止，恢复最优权重
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # 模型保存：保存验证集精度最高的模型
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(PROCESSED_DIR, "best_tomato_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # 学习率调度：验证集损失不下降则降低学习率
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard日志（方便可视化训练过程）
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(ROOT_DIR, "logs", "tomato_model"),
            histogram_freq=1
        )
    ]
    
    # 4. 基础训练（冻结预训练层）
    print("\n🚀 开始基础训练（冻结预训练层）...")
    history_base = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. 模型微调（解冻预训练层顶层，提升精度）
    print("\n🔧 开始模型微调（解冻MobileNetV2顶层）...")
    base_model.trainable = True
    # 只解冻顶层20层，底层保留预训练特征
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # 重新编译（降低学习率，避免破坏预训练权重）
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # 继续微调训练
    history_fine = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS + 10,  # 额外训练10轮
        initial_epoch=history_base.epoch[-1],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. 合并训练历史并保存
    all_history = {
        "loss": history_base.history["loss"] + history_fine.history["loss"],
        "val_loss": history_base.history["val_loss"] + history_fine.history["val_loss"],
        "accuracy": history_base.history["accuracy"] + history_fine.history["accuracy"],
        "val_accuracy": history_base.history["val_accuracy"] + history_fine.history["val_accuracy"]
    }
    history_df = pd.DataFrame(all_history)
    history_df.to_csv(os.path.join(PROCESSED_DIR, "train_history.csv"), index=False)
    
    # 7. 最终评估
    print("\n📊 训练完成 | 最终验证集评估：")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"   - 验证集精度：{val_acc:.4f}")
    print(f"   - 验证集损失：{val_loss:.4f}")
    
    # 输出生成文件路径
    print("\n📁 生成文件清单：")
    print(f"   ✅ 最佳模型：{os.path.join(PROCESSED_DIR, 'best_tomato_model.keras')}")
    print(f"   ✅ 训练历史：{os.path.join(PROCESSED_DIR, 'train_history.csv')}")
    print(f"   ✅ TensorBoard日志：{os.path.join(ROOT_DIR, 'logs', 'tomato_model')}")
    
    return model, history_base, history_fine

# ===================== 主函数：启动训练 =====================
if __name__ == "__main__":
    # 前置校验：预处理目录是否存在
    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(
            f"❌ 预处理目录不存在：{PROCESSED_DIR}\n"
            "请先执行数据预处理脚本：python src/data/prepare_tomato_data.py"
        )
    
    # 执行训练
    try:
        model, history_base, history_fine = train_tomato_model()
        print("\n🎉 番茄胁迫识别模型训练全部完成！")
    except Exception as e:
        print(f"\n❌ 训练过程出错：{str(e)}")
        raise  # 抛出异常，便于定位问题