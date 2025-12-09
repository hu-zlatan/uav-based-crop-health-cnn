# ==============================================================================
# Enhanced Model Evaluation GUI Tool (v5.0)
# File: evaluate_model_gui.py
# Path: src/model/evaluate_model_gui.py
# Features: 
# 1. Metrics Summary Page (All Numeric Metrics)
# 2. Model Hash + Creation Date Display
# 3. Prediction Confidence Scores
# 4. Robust Error Handling
# 5. Fixed FLOPs Formatting Issues
# ==============================================================================

import os
import sys
import time
import hashlib
import psutil
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from datetime import datetime

# ===================== Global Configuration =====================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
EVALUATION_DIR = os.path.join(RESULTS_DIR, "model_evaluations")
IMAGE_SIZE = (256, 256)
NUM_SAMPLE_IMAGES = 8
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===================== Utility: Model Metadata Extraction =====================
def calculate_model_hash(model_path):
    """Calculate MD5 hash for model file (return first 6 chars + full hash)"""
    hash_md5 = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    full_hash = hash_md5.hexdigest()
    short_hash = full_hash[:6]  # 前6位Hash
    return short_hash, full_hash

def get_model_creation_date(model_path):
    """Get model file creation date (YYYYMMDD format)"""
    # Try to extract from directory name first (archive format)
    model_dir = os.path.dirname(model_path)
    dir_name = os.path.basename(model_dir)
    
    # Extract date from archive directory name (e.g., train_vis_20251208_225236_680e6f04)
    if "train_vis_" in dir_name or "tomato_model_archive_" in dir_name:
        date_parts = dir_name.split("_")
        for part in date_parts:
            if len(part) == 8 and part.isdigit():
                return part
    
    # Fallback to file creation time
    create_time = os.path.getctime(model_path)
    return datetime.fromtimestamp(create_time).strftime("%Y%m%d")

def format_model_display_name(model_name, model_path):
    """Format model display name with short hash + date (e.g., 8h64cd-20251208)"""
    short_hash, _ = calculate_model_hash(model_path)
    create_date = get_model_creation_date(model_path)
    
    if model_name == "Current Best Model":
        return f"Current Best Model ({short_hash}-{create_date})"
    else:
        # For archive models: extract original name + add hash-date
        base_name = model_name.replace("Archive: ", "")
        return f"Archive: {base_name} ({short_hash}-{create_date})"

# ===================== Utility: Check Duplicate Evaluation =====================
def check_duplicate_evaluation(model_hash):
    """Check if model has been evaluated before"""
    eval_dir = os.path.join(EVALUATION_DIR, model_hash)
    if os.path.exists(eval_dir) and len(os.listdir(eval_dir)) > 0:
        return True, eval_dir
    return False, os.path.join(EVALUATION_DIR, model_hash)

def get_overwrite_confirmation(model_name):
    """Ask user to confirm overwrite existing evaluation"""
    return messagebox.askyesno(
        "Duplicate Evaluation",
        f"Model '{model_name}' has been evaluated before!\n"
        "Do you want to overwrite existing results?"
    )

# ===================== Utility: Load All Evaluated Models Summary =====================
def load_all_evaluated_models_summary():
    """Load summary data for all evaluated models"""
    summary_data = []
    
    if not os.path.exists(EVALUATION_DIR):
        print(f"Evaluation directory not found: {EVALUATION_DIR}")  # 调试信息
        return summary_data
    
    # Iterate through all model evaluation directories
    for model_hash in os.listdir(EVALUATION_DIR):
        eval_dir = os.path.join(EVALUATION_DIR, model_hash)
        metrics_path = os.path.join(eval_dir, "evaluation_metrics.csv")
        class_metrics_path = os.path.join(eval_dir, "class_wise_metrics.csv")
        
        print(f"Checking: {metrics_path}")  # 调试信息
        
        if os.path.exists(metrics_path):
            try:
                # Load main metrics
                metrics_df = pd.read_csv(metrics_path)
                model_name = metrics_df["model_name"].iloc[0]
                
                # 修复Hash提取逻辑（兼容多种命名格式）
                short_hash = model_hash[:6]  # 直接用目录名的前6位（最可靠）
                
                # Calculate average class accuracy
                avg_class_acc = 0
                if os.path.exists(class_metrics_path):
                    class_df = pd.read_csv(class_metrics_path)
                    avg_class_acc = class_df["class_accuracy"].mean() * 100
                
                # ========== 修复FLOPs处理逻辑 ==========
                flops_value = metrics_df['flops'].iloc[0]
                flops_str = "N/A"
                try:
                    # 处理数字类型的FLOPs
                    flops_num = float(flops_value)
                    if flops_num > 0:
                        flops_str = f"{int(flops_num):,}"
                except:
                    # 处理字符串类型（如"Calculation failed"）
                    flops_str = "N/A"
                
                # Format numeric values
                summary_data.append({
                    "model_name": model_name,
                    "short_hash": short_hash,
                    "accuracy": round(float(metrics_df["accuracy"].iloc[0]), 2),
                    "precision": round(float(metrics_df["macro_precision"].iloc[0]), 2),
                    "recall": round(float(metrics_df["macro_recall"].iloc[0]), 2),
                    "f1_score": round(float(metrics_df["macro_f1"].iloc[0]), 2),
                    "avg_class_accuracy": round(avg_class_acc, 2),
                    "total_params": f"{int(float(metrics_df['total_params'].iloc[0])):,}",
                    "trainable_params": f"{int(float(metrics_df['trainable_params'].iloc[0])):,}",
                    "flops": flops_str,  # 使用修复后的FLOPs字符串
                    "avg_inference_time_ms": round(float(metrics_df["avg_inference_time_ms"].iloc[0]), 2),
                    "fps": round(float(metrics_df["fps"].iloc[0]), 2),
                    "mem_usage_mb": round(float(metrics_df["mem_usage_mb"].iloc[0]), 2),
                    "model_size_mb": round(float(metrics_df["model_size_mb"].iloc[0]), 2),
                    "evaluation_time": metrics_df["evaluation_time"].iloc[0]
                })
                print(f"Loaded data for: {model_name}")  # 调试信息
            except Exception as e:
                print(f"Error loading {metrics_path}: {str(e)}")  # 调试信息
                continue
    
    print(f"Total summary data loaded: {len(summary_data)}")  # 调试信息
    return summary_data

# ===================== Utility: Load Validation Data =====================
def load_validation_data():
    """Load validation dataset, label mapping and training history"""
    # Load label mapping
    label_to_idx_path = os.path.join(PROCESSED_DIR, "label_to_idx.npy")
    if not os.path.exists(label_to_idx_path):
        messagebox.showerror("Error", f"Label mapping file not found:\n{label_to_idx_path}")
        return None, None, None, None, None, None
    
    label_to_idx = np.load(label_to_idx_path, allow_pickle=True).item()
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    
    # Load validation CSV
    val_csv_path = os.path.join(PROCESSED_DIR, "val.csv")
    if not os.path.exists(val_csv_path):
        messagebox.showerror("Error", f"Validation CSV not found:\n{val_csv_path}")
        return None, None, None, None, None, None
    
    val_df = pd.read_csv(val_csv_path)
    
    # Load validation images
    val_images = []
    val_labels = []
    val_img_paths = []
    
    for idx, row in val_df.iterrows():
        img_path = os.path.join(ROOT_DIR, row["img_path"])
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0
            val_images.append(img_array)
            val_labels.append(row["label"])
            val_img_paths.append(img_path)
    
    val_images = np.array(val_images, dtype=np.float32)
    val_labels = np.array(val_labels, dtype=np.int32)
    
    # Load training history (for loss curves)
    history_path = os.path.join(PROCESSED_DIR, "train_history.csv")
    train_history = pd.read_csv(history_path) if os.path.exists(history_path) else None
    
    return val_images, val_labels, idx_to_label, val_img_paths, train_history, class_names

# ===================== Utility: Get Model List =====================
def get_available_models():
    """Get list of available models with formatted display names"""
    model_files = []
    model_metadata = []  # Store (display_name, path, short_hash, full_hash)
    
    # Add best_tomato_model.keras
    best_model_path = os.path.join(PROCESSED_DIR, "best_tomato_model.keras")
    if os.path.exists(best_model_path):
        display_name = format_model_display_name("Current Best Model", best_model_path)
        short_hash, full_hash = calculate_model_hash(best_model_path)
        model_files.append((display_name, best_model_path))
        model_metadata.append((display_name, best_model_path, short_hash, full_hash))
    
    # Add models from results directory
    if os.path.exists(RESULTS_DIR):
        for root, dirs, files in os.walk(RESULTS_DIR):
            for file in files:
                if file.endswith(".keras") and "best_tomato_model" in file:
                    model_path = os.path.join(root, file)
                    dir_name = os.path.basename(os.path.dirname(model_path))
                    original_name = f"Archive: {dir_name}"
                    display_name = format_model_display_name(original_name, model_path)
                    short_hash, full_hash = calculate_model_hash(model_path)
                    model_files.append((display_name, model_path))
                    model_metadata.append((display_name, model_path, short_hash, full_hash))
    
    return model_files, model_metadata

# ===================== Utility: Calculate Advanced Metrics =====================
def calculate_classification_metrics(true_labels, pred_labels, num_classes):
    """Calculate precision, recall, F1-score, class accuracy"""
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    class_accuracy = np.zeros(num_classes)
    
    for cls in range(num_classes):
        tp = np.sum((true_labels == cls) & (pred_labels == cls))
        fp = np.sum((true_labels != cls) & (pred_labels == cls))
        fn = np.sum((true_labels == cls) & (pred_labels != cls))
        tn = np.sum((true_labels != cls) & (pred_labels != cls))
        
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) > 0 else 0
        class_accuracy[cls] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_accuracy": class_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }

def calculate_flops(model):
    """Calculate FLOPs for the model"""
    try:
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_shape = [1] + list(input_shape)[1:]
        
        concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
            tf.TensorSpec(input_shape, model.inputs[0].dtype)
        )
        
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph = frozen_func.graph
        
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        
        return flops.total_float_ops if flops else "Calculation failed"
    except:
        return "Calculation failed"

def calculate_model_size(model_path):
    """Calculate model file size in MB/GB"""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    return size_bytes, size_mb, size_gb

# ===================== Utility: Evaluate Model =====================
def evaluate_model(model_path, val_images, val_labels, class_names):
    """Comprehensive model evaluation with all metrics (including confidence)"""
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        num_classes = len(class_names)
        
        # 1. Memory usage measurement
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        gpu_mem_before = tf.config.experimental.get_memory_info('GPU:0')['used'] / 1024 / 1024 if tf.config.list_physical_devices('GPU') else 0
        
        # 2. Inference speed measurement (with confidence scores)
        model.predict(val_images[:10], verbose=0)  # Warm up
        
        start_time = time.time()
        predictions = model.predict(val_images, verbose=0)  # Raw probability scores
        inference_time = time.time() - start_time
        
        # Calculate confidence scores (max probability)
        confidence_scores = np.max(predictions, axis=1)
        pred_labels = np.argmax(predictions, axis=1)
        
        avg_inference_time = inference_time / len(val_images) * 1000  # ms per image
        fps = len(val_images) / inference_time  # frames per second
        
        # 3. Memory after inference
        mem_after = process.memory_info().rss / 1024 / 1024
        gpu_mem_after = tf.config.experimental.get_memory_info('GPU:0')['used'] / 1024 / 1024 if tf.config.list_physical_devices('GPU') else 0
        mem_usage = mem_after - mem_before
        gpu_mem_usage = gpu_mem_after - gpu_mem_before if tf.config.list_physical_devices('GPU') else 0
        
        # 4. Classification metrics
        accuracy = np.mean(pred_labels == val_labels) * 100
        cls_metrics = calculate_classification_metrics(val_labels, pred_labels, num_classes)
        
        # 5. Model parameters
        total_params = model.count_params()
        trainable_params = sum([np.prod(layer.shape) for layer in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # 6. FLOPs and model size
        flops = calculate_flops(model)
        # 增强FLOPs类型判断
        if isinstance(flops, str) or flops == 0:
            flops = "Calculation failed"
        size_bytes, size_mb, size_gb = calculate_model_size(model_path)
        
        # 7. Confusion matrix
        conf_matrix = tf.math.confusion_matrix(val_labels, pred_labels, num_classes=num_classes).numpy()
        
        return {
            "model": model,
            "predictions": pred_labels,
            "confidence_scores": confidence_scores,  # Add confidence scores
            "conf_matrix": conf_matrix,
            # Basic metrics
            "accuracy": accuracy,
            # Advanced classification metrics
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
            "f1": cls_metrics["f1"],
            "class_accuracy": cls_metrics["class_accuracy"],
            "macro_precision": cls_metrics["macro_precision"] * 100,
            "macro_recall": cls_metrics["macro_recall"] * 100,
            "macro_f1": cls_metrics["macro_f1"] * 100,
            # Efficiency metrics
            "inference_time": inference_time,
            "avg_inference_time": avg_inference_time,
            "fps": fps,
            "mem_usage": mem_usage,
            "gpu_mem_usage": gpu_mem_usage,
            "mem_before": mem_before,
            "mem_after": mem_after,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "flops": flops,
            "model_size_bytes": size_bytes,
            "model_size_mb": size_mb,
            "model_size_gb": size_gb
        }
    
    except Exception as e:
        messagebox.showerror("Evaluation Error", f"Failed to evaluate model:\n{str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ===================== Utility: Visualization Functions (with Confidence) =====================
def plot_confusion_matrix(canvas, conf_matrix, class_names):
    """Plot confusion matrix heatmap"""
    try:
        # 清空画布
        for widget in canvas.winfo_children():
            widget.destroy()
        
        # 缩短类别名（防止重叠）
        short_class_names = [name[:10] + "..." if len(name) > 10 else name for name in class_names]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=short_class_names, yticklabels=short_class_names, ax=ax)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图片
        plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
        return fig
    except Exception as e:
        raise Exception(f"Plot confusion matrix error: {str(e)}")

def plot_loss_curves(canvas, train_history):
    """Plot training/validation loss and accuracy curves"""
    # 清空画布
    for widget in canvas.winfo_children():
        widget.destroy()
    
    if train_history is None or train_history.empty:
        # 显示无数据提示
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, "No training history available", ha="center", va="center", fontsize=14)
        ax.axis('off')
        plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
        return None
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Loss curve
        ax1.plot(train_history["loss"], label="Training Loss", linewidth=2)
        ax1.plot(train_history["val_loss"], label="Validation Loss", linewidth=2)
        ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(train_history["accuracy"], label="Training Accuracy", linewidth=2)
        ax2.plot(train_history["val_accuracy"], label="Validation Accuracy", linewidth=2)
        ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图片
        plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
        return fig
    except Exception as e:
        raise Exception(f"Plot loss curves error: {str(e)}")

def plot_sample_results(canvas, val_images, val_img_paths, val_labels, pred_labels, confidence_scores, idx_to_label):
    """Display sample prediction results WITH CONFIDENCE SCORES"""
    try:
        # 确保样本数量不超过可用数据
        sample_count = min(NUM_SAMPLE_IMAGES, len(val_images))
        sample_indices = np.random.choice(len(val_images), sample_count, replace=False)
        
        # 适配不同的子图布局（2行4列 → 动态行数）
        rows = 2 if sample_count >=4 else 1
        cols = min(4, sample_count)
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
        fig.suptitle("Model Prediction Results (8 Random Samples) - With Confidence Scores", fontsize=16, fontweight="bold")
        
        # 处理单行列的情况
        if sample_count == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()
        
        # 清空画布
        for widget in canvas.winfo_children():
            widget.destroy()
        
        # 绘制样本
        for i, idx in enumerate(sample_indices):
            if i >= len(axes):
                break  # 防止索引越界
            axes[i].imshow(val_images[idx])
            axes[i].axis("off")
            
            # Get labels and confidence
            true_label = idx_to_label[val_labels[idx]]
            pred_label = idx_to_label[pred_labels[idx]]
            confidence = confidence_scores[idx] * 100  # Convert to percentage
            color = "green" if true_label == pred_label else "red"
            
            # Set title with confidence score
            axes[i].set_title(
                f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%",
                color=color, fontsize=10
            )
        
        # 隐藏未使用的子图
        for i in range(len(sample_indices), len(axes)):
            axes[i].axis('off')
        
        # 显示图片
        plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
        return fig
    except Exception as e:
        raise Exception(f"Plot sample error: {str(e)}")

def plot_class_metrics(canvas, metrics, class_names):
    """Plot class-wise precision, recall, F1-score"""
    try:
        # 清空画布
        for widget in canvas.winfo_children():
            widget.destroy()
        
        # 缩短类别名
        short_class_names = [name[:10] + "..." if len(name) > 10 else name for name in class_names]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.bar(range(len(class_names)), metrics["precision"], color="skyblue")
        ax1.set_title("Class-wise Precision", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Class", fontsize=12)
        ax1.set_ylabel("Precision", fontsize=12)
        ax1.set_xticks(range(len(class_names)))
        ax1.set_xticklabels(short_class_names, rotation=45, ha="right")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(range(len(class_names)), metrics["recall"], color="lightgreen")
        ax2.set_title("Class-wise Recall", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Class", fontsize=12)
        ax2.set_ylabel("Recall", fontsize=12)
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(short_class_names, rotation=45, ha="right")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        ax3.bar(range(len(class_names)), metrics["f1"], color="salmon")
        ax3.set_title("Class-wise F1-Score", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Class", fontsize=12)
        ax3.set_ylabel("F1-Score", fontsize=12)
        ax3.set_xticks(range(len(class_names)))
        ax3.set_xticklabels(short_class_names, rotation=45, ha="right")
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图片
        plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
        return fig
    except Exception as e:
        raise Exception(f"Plot class metrics error: {str(e)}")

# ===================== Utility: Save Results =====================
def save_evaluation_results(model_name, model_path, eval_results, train_history, class_names, save_dir):
    """Save all evaluation results to files (including confidence)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Save numerical metrics to CSV
    metrics_df = pd.DataFrame({
        "model_name": [model_name],
        "model_path": [model_path],
        "evaluation_time": [time.strftime('%Y-%m-%d %H:%M:%S')],
        "accuracy": [eval_results["accuracy"]],
        "macro_precision": [eval_results["macro_precision"]],
        "macro_recall": [eval_results["macro_recall"]],
        "macro_f1": [eval_results["macro_f1"]],
        "total_inference_time_s": [eval_results["inference_time"]],
        "avg_inference_time_ms": [eval_results["avg_inference_time"]],
        "fps": [eval_results["fps"]],
        "mem_usage_mb": [eval_results["mem_usage"]],
        "gpu_mem_usage_mb": [eval_results["gpu_mem_usage"]],
        "total_params": [eval_results["total_params"]],
        "trainable_params": [eval_results["trainable_params"]],
        "non_trainable_params": [eval_results["non_trainable_params"]],
        "flops": [eval_results["flops"]],
        "model_size_mb": [eval_results["model_size_mb"]],
        "model_size_gb": [eval_results["model_size_gb"]]
    })
    metrics_df.to_csv(os.path.join(save_dir, "evaluation_metrics.csv"), index=False)
    
    # 2. Save class-wise metrics
    class_metrics_df = pd.DataFrame({
        "class_name": class_names,
        "precision": eval_results["precision"],
        "recall": eval_results["recall"],
        "f1_score": eval_results["f1"],
        "class_accuracy": eval_results["class_accuracy"]
    })
    class_metrics_df.to_csv(os.path.join(save_dir, "class_wise_metrics.csv"), index=False)
    
    # 3. Save confusion matrix
    np.save(os.path.join(save_dir, "confusion_matrix.npy"), eval_results["conf_matrix"])
    
    # 4. Save predictions with confidence scores
    pred_df = pd.DataFrame({
        "true_label": eval_results["predictions"],
        "pred_label": eval_results["predictions"],
        "confidence_score": eval_results["confidence_scores"]  # Add confidence
    })
    pred_df.to_csv(os.path.join(save_dir, "predictions_with_confidence.csv"), index=False)
    
    # 5. Save plots as PNG
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(eval_results["conf_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Loss curves (if available)
    if train_history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.plot(train_history["loss"], label="Training Loss")
        ax1.plot(train_history["val_loss"], label="Validation Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax2.plot(train_history["accuracy"], label="Training Accuracy")
        ax2.plot(train_history["val_accuracy"], label="Validation Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    # Class-wise metrics
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.bar(range(len(class_names)), eval_results["precision"], color="skyblue")
    ax1.set_title("Class-wise Precision")
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha="right")
    ax2.bar(range(len(class_names)), eval_results["recall"], color="lightgreen")
    ax2.set_title("Class-wise Recall")
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha="right")
    ax3.bar(range(len(class_names)), eval_results["f1"], color="salmon")
    ax3.set_title("Class-wise F1-Score")
    ax3.set_xticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    return save_dir

# ===================== GUI: Summary Page =====================
def create_summary_page(notebook):
    """Create summary page with all numeric metrics in table format"""
    summary_frame = ttk.Frame(notebook, padding="10")
    notebook.add(summary_frame, text="Metrics Summary (All Models)")
    
    # Refresh button
    refresh_btn = ttk.Button(summary_frame, text="Refresh Summary", command=lambda: update_summary_table(tree))
    refresh_btn.pack(anchor=tk.W, pady=(0, 10))
    
    # Create treeview for summary table
    columns = [
        "Model Name", "Hash", "Accuracy (%)", "Precision (%)", "Recall (%)", 
        "F1-Score (%)", "Avg Class Acc (%)", "Total Params", "Trainable Params",
        "FLOPs", "Infer Time (ms/img)", "FPS", "Mem Usage (MB)", "Model Size (MB)", "Eval Time"
    ]
    tree = ttk.Treeview(summary_frame, columns=columns, show="headings", height=10)
    
    # Set column headings and widths
    column_widths = [
        200, 60, 80, 80, 80, 80, 80, 120, 120,
        150, 100, 80, 80, 80, 150
    ]
    for i, col in enumerate(columns):
        tree.heading(col, text=col)
        tree.column(col, width=column_widths[i], anchor=tk.CENTER)
    
    # Add scrollbars
    v_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=tree.yview)
    h_scroll = ttk.Scrollbar(summary_frame, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
    
    # Layout
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Initial update
    update_summary_table(tree)
    
    return summary_frame

def update_summary_table(tree):
    """Update summary table with latest data"""
    # Clear existing entries
    for item in tree.get_children():
        tree.delete(item)
    
    # Load summary data
    summary_data = load_all_evaluated_models_summary()
    
    if not summary_data:
        # 显示更详细的提示
        tree.insert("", tk.END, values=[
            "No models evaluated yet! Please:", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            "1. Go to 'Model Selection' tab", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            "2. Select a model from dropdown", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            "3. Click 'Run Comprehensive Evaluation'", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            "4. Wait for evaluation to complete", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
        ])
        return
    
    # Add rows to treeview
    for data in summary_data:
        values = [
            # 限制模型名长度，避免表格变形
            (data["model_name"][:40] + "...") if len(data["model_name"]) > 40 else data["model_name"],
            data["short_hash"],
            data["accuracy"],
            data["precision"],
            data["recall"],
            data["f1_score"],
            data["avg_class_accuracy"],
            data["total_params"],
            data["trainable_params"],
            data["flops"],
            data["avg_inference_time_ms"],
            data["fps"],
            data["mem_usage_mb"],
            data["model_size_mb"],
            data["evaluation_time"]
        ]
        tree.insert("", tk.END, values=values)

# ===================== GUI Main Class =====================
class ModelEvaluationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomato Stress Classification - Advanced Model Evaluation Tool (v5.0)")
        self.root.geometry("1600x900")
        
        # Load validation data
        self.val_images, self.val_labels, self.idx_to_label, self.val_img_paths, self.train_history, self.class_names = load_validation_data()
        if self.val_images is None:
            root.quit()
            return
        
        # Get available models with metadata
        self.model_files, self.model_metadata = get_available_models()
        if not self.model_files:
            messagebox.showerror("Error", "No model files found!")
            root.quit()
            return
        
        # Create GUI components
        self.create_widgets()
    
    def create_widgets(self):
        # Main notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Model Selection Tab
        self.select_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.select_frame, text="Model Selection")
        
        # Model selection combo (with Hash+Date)
        ttk.Label(self.select_frame, text="Select Model (Hash-Date):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.model_var = tk.StringVar()
        model_names = [name for name, path in self.model_files]
        self.model_combo = ttk.Combobox(self.select_frame, textvariable=self.model_var, values=model_names, state="readonly", width=100)
        self.model_combo.current(0)
        self.model_combo.pack(anchor=tk.W, pady=(0, 10))
        
        # Evaluate button
        self.eval_btn = ttk.Button(self.select_frame, text="Run Comprehensive Evaluation", command=self.run_evaluation)
        self.eval_btn.pack(anchor=tk.W, pady=(0, 20))
        
        # Evaluation results text
        ttk.Label(self.select_frame, text="Evaluation Log:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.results_text = tk.Text(self.select_frame, height=20, width=120)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # 2. Metrics Summary Tab (NEW)
        self.summary_frame = create_summary_page(self.notebook)
        
        # 3. Sample Predictions Tab (with Confidence)
        self.sample_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.sample_frame, text="Sample Predictions (with Confidence)")
        self.sample_canvas = ttk.Frame(self.sample_frame)
        self.sample_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 4. Confusion Matrix Tab
        self.conf_matrix_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.conf_matrix_frame, text="Confusion Matrix")
        self.conf_matrix_canvas = ttk.Frame(self.conf_matrix_frame)
        self.conf_matrix_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 5. Loss Curves Tab
        self.loss_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.loss_frame, text="Training/Validation Curves")
        self.loss_canvas = ttk.Frame(self.loss_frame)
        self.loss_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 6. Class-wise Metrics Tab
        self.class_metrics_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.class_metrics_frame, text="Class-wise Metrics")
        self.class_metrics_canvas = ttk.Frame(self.class_metrics_frame)
        self.class_metrics_canvas.pack(fill=tk.BOTH, expand=True)
    
    def run_evaluation(self):
        """Run full model evaluation workflow"""
        try:  # 新增外层异常捕获
            # Get selected model metadata
            selected_idx = self.model_combo.current()
            display_name, model_path = self.model_files[selected_idx]
            _, _, short_hash, full_hash = self.model_metadata[selected_idx]
            
            # Check for duplicate evaluation
            is_duplicate, save_dir = check_duplicate_evaluation(full_hash)
            
            if is_duplicate and not get_overwrite_confirmation(display_name):
                self.results_text.insert(tk.END, "Evaluation cancelled by user.\n")
                return
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Starting evaluation for: {display_name}\n")
            self.results_text.insert(tk.END, f"Model path: {model_path}\n")
            self.results_text.insert(tk.END, f"Model short hash: {short_hash} | Full hash: {full_hash}\n")
            self.results_text.insert(tk.END, "Please wait - this may take several minutes...\n")
            self.root.update_idletasks()
            
            # Run evaluation
            eval_results = evaluate_model(model_path, self.val_images, self.val_labels, self.class_names)
            if eval_results is None:
                return
            
            # Generate visualizations (with confidence)
            self.results_text.insert(tk.END, "Generating visualizations with confidence scores...\n")
            self.root.update_idletasks()
            
            # ========== 为每个可视化函数添加异常捕获 ==========
            try:
                # Plot sample results WITH CONFIDENCE
                plot_sample_results(
                    self.sample_canvas, self.val_images, self.val_img_paths,
                    self.val_labels, eval_results["predictions"],
                    eval_results["confidence_scores"], self.idx_to_label
                )
            except Exception as e:
                self.results_text.insert(tk.END, f"\n⚠️ Error plotting sample results: {str(e)}\n")
                import traceback
                self.results_text.insert(tk.END, f"Traceback: {traceback.format_exc()}\n")
            
            try:
                # Plot confusion matrix
                plot_confusion_matrix(self.conf_matrix_canvas, eval_results["conf_matrix"], self.class_names)
            except Exception as e:
                self.results_text.insert(tk.END, f"\n⚠️ Error plotting confusion matrix: {str(e)}\n")
                import traceback
                self.results_text.insert(tk.END, f"Traceback: {traceback.format_exc()}\n")
            
            try:
                # Plot loss curves
                plot_loss_curves(self.loss_canvas, self.train_history)
            except Exception as e:
                self.results_text.insert(tk.END, f"\n⚠️ Error plotting loss curves: {str(e)}\n")
                import traceback
                self.results_text.insert(tk.END, f"Traceback: {traceback.format_exc()}\n")
            
            try:
                # Plot class-wise metrics
                plot_class_metrics(self.class_metrics_canvas, eval_results, self.class_names)
            except Exception as e:
                self.results_text.insert(tk.END, f"\n⚠️ Error plotting class metrics: {str(e)}\n")
                import traceback
                self.results_text.insert(tk.END, f"Traceback: {traceback.format_exc()}\n")
            
            # Calculate average confidence score
            avg_confidence = np.mean(eval_results["confidence_scores"]) * 100
            
            # 先处理FLOPs的显示文本（单独提取，避免f-string嵌套错误）
            flops_display = ""
            if eval_results['flops'] != "Calculation failed":
                try:
                    flops_display = f"{int(eval_results['flops']):,} operations"
                except:
                    flops_display = "Calculation failed"
            else:
                flops_display = "Calculation failed"
            
            # Generate detailed results text
            results_text = f"""
========================================
COMPREHENSIVE MODEL EVALUATION RESULTS
========================================
Model Information:
  - Model Name: {display_name}
  - Model Path: {model_path}
  - Model Short Hash: {short_hash} | Full Hash: {full_hash}
  - Evaluation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

----------------------------------------
CORE CLASSIFICATION METRICS:
  - Overall Accuracy: {eval_results['accuracy']:.2f}%
  - Macro Precision: {eval_results['macro_precision']:.2f}%
  - Macro Recall: {eval_results['macro_recall']:.2f}%
  - Macro F1-Score: {eval_results['macro_f1']:.2f}%
  - Average Prediction Confidence: {avg_confidence:.2f}%

----------------------------------------
INFERENCE PERFORMANCE:
  - Total Inference Time: {eval_results['inference_time']:.2f} seconds
  - Average Inference Time: {eval_results['avg_inference_time']:.2f} ms/image
  - Frames Per Second (FPS): {eval_results['fps']:.2f}

----------------------------------------
MEMORY USAGE:
  - Memory Before Inference: {eval_results['mem_before']:.2f} MB
  - Memory After Inference: {eval_results['mem_after']:.2f} MB
  - Additional Memory Used: {eval_results['mem_usage']:.2f} MB
  - GPU Memory Used: {eval_results['gpu_mem_usage']:.2f} MB (if available)

----------------------------------------
MODEL SIZE & COMPLEXITY:
  - Total Parameters: {eval_results['total_params']:,}
  - Trainable Parameters: {eval_results['trainable_params']:,}
  - Non-trainable Parameters: {eval_results['non_trainable_params']:,}
  - FLOPs: {flops_display}
  - Model File Size: {eval_results['model_size_mb']:.2f} MB ({eval_results['model_size_gb']:.4f} GB)

----------------------------------------
CLASS-WISE ACCURACY (Top 5):
"""
            
            # Add top 5 class accuracies
            class_acc = eval_results["class_accuracy"] * 100
            top_classes = np.argsort(class_acc)[-5:][::-1]
            for i, cls_idx in enumerate(top_classes):
                results_text += f"  {i+1}. {self.class_names[cls_idx]}: {class_acc[cls_idx]:.2f}%\n"
            
            results_text += f"""
========================================
RESULTS SAVED TO: {save_dir}
========================================
"""
            
            # Update results text
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_text)
            
            # ========== 保存结果时添加异常捕获 ==========
            try:
                # Save all results (including confidence)
                save_path = save_evaluation_results(
                    display_name, model_path, eval_results, 
                    self.train_history, self.class_names, save_dir
                )
                self.results_text.insert(tk.END, f"\n✅ All results (including confidence scores) saved to:\n{save_path}")
            except Exception as e:
                self.results_text.insert(tk.END, f"\n⚠️ Error saving results: {str(e)}\n")
                import traceback
                self.results_text.insert(tk.END, f"Traceback: {traceback.format_exc()}\n")
            
            # ========== 刷新汇总表时添加异常捕获 ==========
            try:
                # Refresh summary table
                for child in self.summary_frame.winfo_children():
                    if isinstance(child, ttk.Treeview):
                        update_summary_table(child)
            except Exception as e:
                self.results_text.insert(tk.END, f"\n⚠️ Error refreshing summary table: {str(e)}\n")
                import traceback
                self.results_text.insert(tk.END, f"Traceback: {traceback.format_exc()}\n")
            
            # ========== 确保弹窗能显示 ==========
            messagebox.showinfo("Success", "Evaluation completed successfully!\nAll results (including confidence scores) have been saved.\nSummary table has been updated.")
            
        except Exception as e:  # 捕获所有未处理的异常
            self.results_text.insert(tk.END, f"\n❌ Critical error during evaluation: {str(e)}\n")
            import traceback
            self.results_text.insert(tk.END, f"Full traceback:\n{traceback.format_exc()}\n")
            messagebox.showerror("Error", f"Evaluation failed:\n{str(e)}")

# ===================== Main Function =====================
def main():
    root = tk.Tk()
    app = ModelEvaluationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()