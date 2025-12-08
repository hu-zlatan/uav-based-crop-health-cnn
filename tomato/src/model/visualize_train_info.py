# ==============================================================================
# Model Training Information Visualization Script
# File: visualize_train_info.py
# Path: src/model/visualize_train_info.py
# Function: Plot training curves (Loss/Accuracy) + Summarize model info + Archive model file
# Commit: v1.1.0 (2025-12-08) - English only output + model file archiving
# ==============================================================================

import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

# ===================== Core Configuration =====================
# Project root directory (adapt to src/model/ path)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Processed data/model directory
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
# Visualization output directory (auto-create with timestamp)
VIS_OUTPUT_DIR = os.path.join(
    ROOT_DIR, "results", 
    f"train_vis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
# Image configuration
IMG_FORMAT = "png"
IMG_DPI = 300  # Image resolution
# Set matplotlib to English only (remove Chinese font config)
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== Utility: Create Directories =====================
def create_dirs():
    """Create output directories for visualization results"""
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(VIS_OUTPUT_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(VIS_OUTPUT_DIR, "model"), exist_ok=True)  # For model file
    print(f"✅ Visualization archive directory: {VIS_OUTPUT_DIR}")

# ===================== Utility: Load Training Data =====================
def load_train_data():
    """Load training history, label mapping, and model"""
    # 1. Load training history
    history_path = os.path.join(PROCESSED_DIR, "train_history.csv")
    if not os.path.exists(history_path):
        history_path = os.path.join(PROCESSED_DIR, "train_history_v1.0.0.csv")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"❌ Training history file not found: {history_path}")
    history_df = pd.read_csv(history_path)
    print(f"✅ Loaded training history: {history_df.shape} records")

    # 2. Load label mapping
    label_to_idx_path = os.path.join(PROCESSED_DIR, "label_to_idx.npy")
    if not os.path.exists(label_to_idx_path):
        raise FileNotFoundError(f"❌ Label mapping file not found: {label_to_idx_path}")
    label_to_idx = np.load(label_to_idx_path, allow_pickle=True).item()
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    print(f"✅ Loaded label mapping: {len(label_to_idx)} classes")

    # 3. Load model
    model_path = os.path.join(PROCESSED_DIR, "best_tomato_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Loaded model: {model.name}")

    return history_df, label_to_idx, idx_to_label, model, model_path

# ===================== Utility: Plot Training Curves =====================
def plot_train_curves(history_df):
    """
    Plot training curves:
    1. Loss curve (train + validation)
    2. Accuracy curve (train + validation)
    """
    # Create 2x1 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Tomato Stress Classification - Training Curves", fontsize=16, fontweight="bold")

    # 1. Plot Loss curve
    epochs = range(1, len(history_df["loss"]) + 1)
    ax1.plot(epochs, history_df["loss"], "b-", linewidth=2, label="Training Loss")
    ax1.plot(epochs, history_df["val_loss"], "r-", linewidth=2, label="Validation Loss")
    ax1.set_title("Training & Validation Loss", fontsize=14)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss Value", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Annotate minimum loss values
    min_train_loss = history_df["loss"].min()
    min_val_loss = history_df["val_loss"].min()
    ax1.annotate(f"Min: {min_train_loss:.4f}", 
                xy=(history_df["loss"].idxmin()+1, min_train_loss),
                xytext=(history_df["loss"].idxmin()+1, min_train_loss+0.1),
                arrowprops=dict(arrowstyle="->", color="blue"))
    ax1.annotate(f"Min: {min_val_loss:.4f}", 
                xy=(history_df["val_loss"].idxmin()+1, min_val_loss),
                xytext=(history_df["val_loss"].idxmin()+1, min_val_loss+0.1),
                arrowprops=dict(arrowstyle="->", color="red"))

    # 2. Plot Accuracy curve
    ax2.plot(epochs, history_df["accuracy"], "b-", linewidth=2, label="Training Accuracy")
    ax2.plot(epochs, history_df["val_accuracy"], "r-", linewidth=2, label="Validation Accuracy")
    ax2.set_title("Training & Validation Accuracy", fontsize=14)
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Annotate maximum accuracy values
    max_train_acc = history_df["accuracy"].max()
    max_val_acc = history_df["val_accuracy"].max()
    ax2.annotate(f"Max: {max_train_acc:.4f}", 
                xy=(history_df["accuracy"].idxmax()+1, max_train_acc),
                xytext=(history_df["accuracy"].idxmax()+1, max_train_acc-0.05),
                arrowprops=dict(arrowstyle="->", color="blue"))
    ax2.annotate(f"Max: {max_val_acc:.4f}", 
                xy=(history_df["val_accuracy"].idxmax()+1, max_val_acc),
                xytext=(history_df["val_accuracy"].idxmax()+1, max_val_acc-0.05),
                arrowprops=dict(arrowstyle="->", color="red"))

    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(VIS_OUTPUT_DIR, "plots", f"train_curves.{IMG_FORMAT}")
    plt.savefig(plot_path, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ Training curves saved to: {plot_path}")

    # Return key metrics
    key_metrics = {
        "max_train_accuracy": float(max_train_acc),
        "max_val_accuracy": float(max_val_acc),
        "min_train_loss": float(min_train_loss),
        "min_val_loss": float(min_val_loss),
        "best_epoch_val_acc": int(history_df["val_accuracy"].idxmax() + 1),
        "total_epochs": len(epochs)
    }
    return key_metrics

# ===================== Utility: Archive Model File =====================
def archive_model_file(model_source_path):
    """Copy .keras model file to archive directory"""
    model_dest_path = os.path.join(VIS_OUTPUT_DIR, "model", "best_tomato_model.keras")
    shutil.copy2(model_source_path, model_dest_path)  # Preserve file metadata
    print(f"✅ Model file archived to: {model_dest_path}")
    return model_dest_path

# ===================== Utility: Summarize Model Info =====================
def summarize_model_info(model, label_to_idx, key_metrics, model_archive_path):
    """Summarize core model information and save as JSON/text"""
    # 1. Get model summary as text
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary_text = "\n".join(model_summary)

    # 2. Core model information dictionary
    model_info = {
        "basic_info": {
            "model_name": model.name,
            "total_parameters": model.count_params(),
            "trainable_parameters": sum([np.prod(layer.shape) for layer in model.trainable_weights]),
            "non_trainable_parameters": sum([np.prod(layer.shape) for layer in model.non_trainable_weights]),
            "input_shape": model.input_shape[0],
            "output_shape": model.output_shape[0],
            "archive_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_archive_path": model_archive_path
        },
        "training_metrics": key_metrics,
        "dataset_info": {
            "number_of_classes": len(label_to_idx),
            "label_mapping": label_to_idx
        },
        "training_configuration": {
            "batch_size": 32,
            "base_learning_rate": 1e-4,
            "fine_tune_learning_rate": 1e-5,
            "image_size": (256, 256)
        }
    }

    # 3. Save model info as JSON
    info_path = os.path.join(VIS_OUTPUT_DIR, "model_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)
    print(f"✅ Model information saved to: {info_path}")

    # 4. Save model summary as text
    summary_path = os.path.join(VIS_OUTPUT_DIR, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(model_summary_text)
    print(f"✅ Model summary saved to: {summary_path}")

    return model_info

# ===================== Main Function =====================
def main():
    """Main workflow: Load data → Plot curves → Archive model → Summarize info"""
    print("🚀 Starting model training information visualization...")
    
    # 1. Create output directories
    create_dirs()

    # 2. Load training data and model
    history_df, label_to_idx, idx_to_label, model, model_source_path = load_train_data()

    # 3. Plot training curves
    key_metrics = plot_train_curves(history_df)

    # 4. Archive model file
    model_archive_path = archive_model_file(model_source_path)

    # 5. Summarize and save model information
    model_info = summarize_model_info(model, label_to_idx, key_metrics, model_archive_path)

    # 6. Print final summary
    print("\n📊 Model Training Core Metrics Summary:")
    print(f"   - Maximum Validation Accuracy: {model_info['training_metrics']['max_val_accuracy']:.4f} (Epoch {model_info['training_metrics']['best_epoch_val_acc']})")
    print(f"   - Minimum Validation Loss: {model_info['training_metrics']['min_val_loss']:.4f}")
    print(f"   - Total Model Parameters: {model_info['basic_info']['total_parameters']:,}")
    print(f"   - Trainable Parameters: {model_info['basic_info']['trainable_parameters']:,}")
    print(f"   - Number of Classes: {model_info['dataset_info']['number_of_classes']}")

    print(f"\n🎉 All visualization results archived to: {VIS_OUTPUT_DIR}")
    print(f"   - Training curves: plots/train_curves.{IMG_FORMAT}")
    print(f"   - Model file: model/best_tomato_model.keras")
    print(f"   - Model information: model_info.json")
    print(f"   - Model summary: model_summary.txt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during visualization: {str(e)}")
        raise