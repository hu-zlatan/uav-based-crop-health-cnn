# ==============================================================================
# Model Training Information Visualization Script
# File: visualize_train_info.py
# Path: src/model/visualize_train_info.py
# Function: Plot training curves + Summarize model info + Archive model + Duplicate protection
# Commit: v2.0.0 (2025-12-08) - Fix duplicate archive issue (hash-based unique directory)
# ==============================================================================

import os
import json
import shutil
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# ===================== Core Configuration =====================
# Project root directory (adapt to src/model/ path)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
# Image configuration
IMG_FORMAT = "png"
IMG_DPI = 300
# Set matplotlib to English only
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== Utility: Custom JSON Encoder =====================
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return super(NumpyEncoder, self).default(obj)

# ===================== Utility: Model Validation & Duplicate Check =====================
def calculate_model_hash(model_path):
    """Calculate MD5 hash of model file for unique identification"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    
    # Calculate MD5 hash
    hash_md5 = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    model_hash = hash_md5.hexdigest()
    print(f"✅ Model MD5 hash: {model_hash}")
    return model_hash

def get_model_archive_dir(model_hash):
    """Get unique archive directory based on model hash (no timestamp)"""
    # Use full hash for uniqueness (no time suffix)
    archive_dir = os.path.join(RESULTS_DIR, f"tomato_model_archive_{model_hash}")
    return archive_dir

def check_model_archive_exists(model_hash):
    """Check if model archive already exists (hash-based)"""
    archive_dir = get_model_archive_dir(model_hash)
    return os.path.exists(archive_dir)

def get_user_confirmation_for_overwrite(model_hash, model_path):
    """Get user confirmation for overwriting existing archive"""
    archive_dir = get_model_archive_dir(model_hash)
    
    if not os.path.exists(archive_dir):
        return True  # No existing archive, proceed
    
    # Show overwrite warning
    print("\n⚠️  WARNING: Model archive already exists!")
    print(f"   Model file: {model_path}")
    print(f"   Existing archive: {archive_dir}")
    print(f"   Overwriting will replace ALL files in this directory!")
    
    # Get user input
    while True:
        user_input = input("\n❓ Overwrite existing archive (y/n)? ").strip().lower()
        if user_input in ["y", "yes"]:
            # Backup existing files before overwrite (optional)
            backup_dir = f"{archive_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(archive_dir, backup_dir)
            print(f"ℹ️  Backed up existing archive to: {backup_dir}")
            return True
        elif user_input in ["n", "no"]:
            print("ℹ️  Operation cancelled by user")
            return False
        else:
            print("❌ Invalid input! Please enter 'y' or 'n'")

def create_model_archive_dir(model_hash):
    """Create unique archive directory (hash-based, no timestamp)"""
    archive_dir = get_model_archive_dir(model_hash)
    
    # Create directories (overwrite handled by user confirmation)
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(os.path.join(archive_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(archive_dir, "model"), exist_ok=True)
    
    print(f"✅ Archive directory created/updated: {archive_dir}")
    return archive_dir

# ===================== Utility: Load Training Data =====================
def load_train_data(model_path):
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
    label_to_idx = {k: int(v) for k, v in label_to_idx.items()}
    print(f"✅ Loaded label mapping: {len(label_to_idx)} classes")

    # 3. Load model
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Loaded model: {model.name}")

    return history_df, label_to_idx, model

# ===================== Utility: Plot Training Curves =====================
def plot_train_curves(history_df, archive_dir):
    """Plot training curves (Loss/Accuracy) and save to archive"""
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
    plot_path = os.path.join(archive_dir, "plots", f"train_curves.{IMG_FORMAT}")
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
        "total_epochs": len(epochs),
        "plot_generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return key_metrics

# ===================== Utility: Archive Model File =====================
def archive_model_file(model_source_path, archive_dir):
    """Copy .keras model file to hash-based archive directory"""
    model_dest_path = os.path.join(archive_dir, "model", "best_tomato_model.keras")
    
    # Copy model file (overwrite if exists)
    shutil.copy2(model_source_path, model_dest_path)
    print(f"✅ Model file archived to: {model_dest_path}")
    
    return model_dest_path

# ===================== Utility: Summarize Model Info =====================
def summarize_model_info(model, label_to_idx, key_metrics, model_archive_path, model_hash, archive_dir):
    """Summarize core model information with hash for unique identification"""
    # 1. Get model summary as text
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary_text = "\n".join(model_summary)

    # 2. Safe shape conversion (fix NoneType error)
    def convert_shape(shape):
        """Safe conversion of model shape with None handling"""
        if shape is None:
            return None
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()
        if not isinstance(shape, (list, tuple)):
            return int(shape) if shape is not None else None
        return [int(dim) if dim is not None else None for dim in shape]

    # 3. Get safe input/output shapes
    try:
        input_shape = convert_shape(model.input_shape[0] if isinstance(model.input_shape, (list, tuple)) else model.input_shape)
    except:
        input_shape = None
    
    try:
        output_shape = convert_shape(model.output_shape[0] if isinstance(model.output_shape, (list, tuple)) else model.output_shape)
    except:
        output_shape = None

    # 4. Core model information dictionary (with update time)
    model_info = {
        "basic_info": {
            "model_name": model.name,
            "model_hash": model_hash,  # Unique model identifier (no duplicates)
            "total_parameters": int(model.count_params()),
            "trainable_parameters": int(sum([np.prod(layer.shape) for layer in model.trainable_weights])),
            "non_trainable_parameters": int(sum([np.prod(layer.shape) for layer in model.non_trainable_weights])),
            "input_shape": input_shape,
            "output_shape": output_shape,
            "archive_creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_archive_path": model_archive_path,
            "archive_directory": archive_dir
        },
        "training_metrics": key_metrics,
        "dataset_info": {
            "number_of_classes": len(label_to_idx),
            "label_mapping": label_to_idx
        },
        "training_configuration": {
            "batch_size": 32,
            "base_learning_rate": float(1e-4),
            "fine_tune_learning_rate": float(1e-5),
            "image_size": [256, 256]
        }
    }

    # 5. Update existing info if available (preserve history)
    info_path = os.path.join(archive_dir, "model_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                existing_info = json.load(f)
            # Preserve creation time, update last updated time
            model_info["basic_info"]["archive_creation_time"] = existing_info["basic_info"].get("archive_creation_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            # Append training metrics history
            if "training_metrics_history" not in existing_info:
                existing_info["training_metrics_history"] = []
            existing_info["training_metrics_history"].append(key_metrics)
            model_info["training_metrics_history"] = existing_info["training_metrics_history"]
        except:
            pass

    # 6. Save model info with custom JSON encoder
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
    print(f"✅ Model information saved to: {info_path}")

    # 7. Save model summary as text
    summary_path = os.path.join(archive_dir, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(model_summary_text)
    print(f"✅ Model summary saved to: {summary_path}")

    return model_info

# ===================== Main Function =====================
def main():
    """Main workflow with hash-based unique archive (no duplicates)"""
    print("🚀 Starting model training information visualization (v2.0) - Hash-based archive...")
    
    # 1. Define model path
    model_path = os.path.join(PROCESSED_DIR, "best_tomato_model.keras")
    
    # 2. Core duplicate protection logic
    try:
        # Calculate model hash (unique identifier)
        model_hash = calculate_model_hash(model_path)
        
        # Check if archive exists and get user confirmation
        if not get_user_confirmation_for_overwrite(model_hash, model_path):
            return  # Exit if user cancels
        
        # Create unique hash-based archive directory (no timestamp)
        archive_dir = create_model_archive_dir(model_hash)
        
        # 3. Load training data
        history_df, label_to_idx, model = load_train_data(model_path)
        
        # 4. Plot training curves
        key_metrics = plot_train_curves(history_df, archive_dir)
        
        # 5. Archive model file (overwrite with user confirmation)
        model_archive_path = archive_model_file(model_path, archive_dir)
        
        # 6. Summarize and save model information
        model_info = summarize_model_info(model, label_to_idx, key_metrics, model_archive_path, model_hash, archive_dir)
        
        # 7. Print final summary
        print("\n📊 Model Training Core Metrics Summary:")
        print(f"   - Maximum Validation Accuracy: {model_info['training_metrics']['max_val_accuracy']:.4f} (Epoch {model_info['training_metrics']['best_epoch_val_acc']})")
        print(f"   - Minimum Validation Loss: {model_info['training_metrics']['min_val_loss']:.4f}")
        print(f"   - Total Model Parameters: {model_info['basic_info']['total_parameters']:,}")
        print(f"   - Trainable Parameters: {model_info['basic_info']['trainable_parameters']:,}")
        print(f"   - Number of Classes: {model_info['dataset_info']['number_of_classes']}")
        print(f"   - Model Hash: {model_hash}")
        print(f"   - Archive Directory: {archive_dir}")

        print(f"\n🎉 All results archived to hash-based directory: {archive_dir}")
        print(f"   - Training curves: plots/train_curves.{IMG_FORMAT}")
        print(f"   - Model file: model/best_tomato_model.keras")
        print(f"   - Model information: model_info.json")
        print(f"   - Model summary: model_summary.txt")
        
    except Exception as e:
        print(f"\n❌ Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()