import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===================== 核心配置（相对路径 + 无color子文件夹） =====================
# 脚本路径：src/data/prepare_tomato_data.py → 项目根目录/data/raw
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
IMAGE_SIZE = (256, 256)
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_SEED = 42

# ===================== 核心工具函数（解析Tomato___XXX命名） =====================
def parse_tomato_label(folder_name):
    """
    解析Tomato___XXX格式的文件夹名，提取胁迫类型标签
    示例：Tomato___Bacterial_spot → Bacterial_spot
    示例：Tomato___Healthy → Healthy
    """
    pattern = r"Tomato___(.*)"
    match = re.match(pattern, folder_name)
    if match:
        label = match.group(1).replace("_", " ").title()
        return label.strip()
    else:
        return None

def create_dirs():
    """创建必要目录 + 验证原始数据路径"""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"✅ 预处理输出目录就绪：{PROCESSED_DATA_DIR}")
    
    # 验证raw路径是否存在
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"❌ 原始数据路径不存在：{RAW_DATA_DIR}，请确认文件已复制到data/raw")
    print(f"✅ 原始数据路径验证通过：{RAW_DATA_DIR}")

def analyze_tomato_dataset():
    """
    核心：直接遍历data/raw下的Tomato___XXX文件夹，统计每类标签+样本数
    （无color子文件夹，直接处理raw根目录下的番茄文件夹）
    """
    sample_list = []
    class_count = {}  # {标签: 样本数}

    # 直接遍历RAW_DATA_DIR（data/raw）下的所有文件夹
    all_folders = os.listdir(RAW_DATA_DIR)
    for folder_name in tqdm(all_folders, desc="遍历番茄胁迫类别文件夹"):
        folder_path = os.path.join(RAW_DATA_DIR, folder_name)
        
        # 仅处理：文件夹 + 以Tomato___开头
        if not os.path.isdir(folder_path) or not folder_name.startswith("Tomato___"):
            print(f"⚠️  跳过非番茄类别文件夹：{folder_name}")
            continue
        
        # 解析胁迫标签
        stress_label = parse_tomato_label(folder_name)
        if not stress_label:
            print(f"⚠️  跳过无效命名文件夹：{folder_name}")
            continue
        
        # 统计该文件夹下的所有图片
        img_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
        ]
        if not img_files:
            print(f"⚠️  类别{stress_label}下无图片，跳过")
            continue
        
        # 记录类别样本数
        class_count[stress_label] = len(img_files)
        
        # 记录每个样本的详细信息（相对路径+绝对路径）
        for img_file in img_files:
            # 相对路径：data/raw/XXX/XXX.jpg
            relative_img_path = os.path.join("data", "raw", folder_name, img_file)
            # 绝对路径：用于加载图片
            absolute_img_path = os.path.join(folder_path, img_file)
            
            sample_list.append({
                "img_path": relative_img_path,
                "absolute_img_path": absolute_img_path,
                "folder_name": folder_name,
                "stress_label": stress_label,
                "is_healthy": True if stress_label == "Healthy" else False,
                "image_size": IMAGE_SIZE
            })

    # 空数据集校验
    if not sample_list:
        raise ValueError("❌ 未找到任何番茄样本！请确认data/raw下有Tomato___XXX格式的文件夹")
    
    # 转换为DataFrame
    df = pd.DataFrame(sample_list)
    
    # ========== 输出类别-样本数统计 ==========
    print("\n" + "="*50)
    print("🍅 番茄胁迫类别 - 样本数统计（按样本数降序）")
    print("="*50)
    sorted_class_count = dict(sorted(class_count.items(), key=lambda x: x[1], reverse=True))
    for idx, (label, count) in enumerate(sorted_class_count.items(), 1):
        print(f"{idx:2d}. {label:<20} : {count:>5} 张")
    
    # 汇总信息
    total_samples = sum(class_count.values())
    total_classes = len(class_count)
    print("="*50)
    print(f"📊 汇总：共 {total_classes} 个胁迫类别，总计 {total_samples} 张图片")
    print(f"🥬 健康样本数：{class_count.get('Healthy', 0)} 张")
    print("="*50)

    # ========== 保存统计结果到CSV ==========
    class_count_df = pd.DataFrame({
        "stress_label": list(class_count.keys()),
        "sample_count": list(class_count.values()),
        "sample_ratio": [f"{count/total_samples*100:.2f}%" for count in class_count.values()]
    }).sort_values(by="sample_count", ascending=False)
    class_count_csv_path = os.path.join(PROCESSED_DATA_DIR, "class_sample_count.csv")
    class_count_df.to_csv(class_count_csv_path, index=False, encoding="utf-8")
    print(f"\n✅ 类别-样本数统计已保存：{class_count_csv_path}")

    # ========== 可视化类别分布 ==========
    plt.figure(figsize=(18, 8))
    sns.barplot(x=list(sorted_class_count.keys()), y=list(sorted_class_count.values()), palette="viridis")
    # 添加数值标签
    for idx, count in enumerate(sorted_class_count.values()):
        plt.text(idx, count + 5, str(count), ha="center", fontsize=9)
    plt.title("Tomato Stress Class - Sample Count Distribution (256×256)", fontsize=14)
    plt.xlabel("Stress Label", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # 保存图表
    class_dist_png_path = os.path.join(PROCESSED_DATA_DIR, "class_sample_distribution.png")
    plt.savefig(class_dist_png_path, dpi=150)
    plt.close()
    print(f"✅ 类别分布可视化已保存：{class_dist_png_path}")

    return df, class_count

def split_dataset(df, class_count):
    """分层划分训练/验证/测试集（保证每类样本分布均匀）"""
    # 标签→数字编码映射
    stress_labels = sorted(class_count.keys())
    label_to_idx = {label: idx for idx, label in enumerate(stress_labels)}
    df["label"] = df["stress_label"].map(label_to_idx)

    # 分层划分
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["stress_label"], random_state=RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=VAL_SIZE/(1-TEST_SIZE), stratify=train_val_df["stress_label"], random_state=RANDOM_SEED
    )

    # 保存划分结果（仅保留核心列）
    save_cols = ["img_path", "stress_label", "label", "is_healthy"]
    train_df[save_cols].to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False, encoding="utf-8")
    val_df[save_cols].to_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"), index=False, encoding="utf-8")
    test_df[save_cols].to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False, encoding="utf-8")

    # 保存标签映射
    np.save(os.path.join(PROCESSED_DATA_DIR, "label_to_idx.npy"), label_to_idx)
    np.save(os.path.join(PROCESSED_DATA_DIR, "idx_to_label.npy"), {v: k for k, v in label_to_idx.items()})

    # 打印划分结果
    print("\n=== 数据集划分结果 ===")
    print(f"训练集：{len(train_df)} 张 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集：{len(val_df)} 张 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"测试集：{len(test_df)} 张 ({len(test_df)/len(df)*100:.1f}%)")

    # 验证训练集类别分布
    print("\n=== 训练集各类别样本数（前5类）===")
    train_class_count = train_df["stress_label"].value_counts().head()
    for label, count in train_class_count.items():
        print(f"{label:<20} : {count:>5} 张")

    return train_df, val_df, test_df, label_to_idx

def validate_image_preprocessing(df):
    """验证图片预处理（256×256）"""
    sample_df = df.sample(10, random_state=RANDOM_SEED)
    plt.figure(figsize=(18, 10))

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        try:
            # 用绝对路径加载图片
            img = Image.open(row["absolute_img_path"]).convert("RGB")
            img_resized = img.resize(IMAGE_SIZE)
            img_array = np.array(img_resized) / 255.0

            # 校验尺寸
            assert img_array.shape == (256, 256, 3), f"尺寸错误：{img_array.shape}"

            # 提取文件名（无反斜杠）
            img_filename = os.path.basename(row["absolute_img_path"])
            short_filename = img_filename[:10]
            
            # 可视化
            plt.subplot(2, 5, idx+1)
            plt.imshow(img_array)
            plt.title(f"{row['stress_label']}\n({short_filename}...)", fontsize=9)
            plt.axis("off")
        except Exception as e:
            print(f"⚠️  图片{row['absolute_img_path']}处理失败：{e}")
            continue

    plt.suptitle("Sample Preprocessed Tomato Images (256×256)", fontsize=14)
    plt.tight_layout()
    sample_png_path = os.path.join(PROCESSED_DATA_DIR, "sample_images.png")
    plt.savefig(sample_png_path, dpi=150)
    plt.close()
    print(f"✅ 样本预处理验证完成：{sample_png_path}")

# ===================== 主函数 =====================
if __name__ == "__main__":
    create_dirs()
    # 核心：统计类别+样本数
    df, class_count = analyze_tomato_dataset()
    # 划分数据集
    train_df, val_df, test_df, label_to_idx = split_dataset(df, class_count)
    # 验证预处理
    validate_image_preprocessing(train_df)

    print("\n🎯 数据预处理完成！核心输出文件：")
    print(f"  1. {PROCESSED_DATA_DIR}/class_sample_count.csv → 类别-样本数统计")
    print(f"  2. {PROCESSED_DATA_DIR}/class_sample_distribution.png → 类别分布可视化")
    print(f"  3. {PROCESSED_DATA_DIR}/label_to_idx.npy → 标签-数字编码映射")
    print(f"  4. {PROCESSED_DATA_DIR}/train/val/test.csv → 划分后的数据集")