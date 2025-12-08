# ==============================================================================
# Model Training Information Visualization Script
# File: visualize_train_info.py
# Path: src/model/visualize_train_info.py
# Function: Plot training curves + Summarize model info + Archive model + Duplicate check
# Commit: v1.4.0 (2025-12-08) - Fix NoneType iteration error in shape processing
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

def check_existing_archives(model_hash):
    """Check if model already has archive records"""
    existing_archives = []
    if not os.path.exists(RESULTS_DIR):
        return existing_archives
    
    # Scan all train_vis directories
    for archive_dir in Path(RESULTS_DIR).glob("train_vis_*"):
        info_path = archive_dir / "model_info.json"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                # Check if model hash exists in info (for existing archives)
                if "model_hash" in info["basic_info"] and info["basic_info"]["model_hash"] == model_hash:
                    existing_archives.append(str(archive_dir))
            except:
                continue
    return existing_archives

def get_user_confirmation(existing_archives, model_path):
    """Get user confirmation for overwriting duplicate archives"""
    if not existing_archives:
        return True  # No duplicates, proceed
    
    # Show duplicate warning
    print("\n⚠️  WARNING: Duplicate model detected!")
    print(f"   Model file: {model_path}")
    print(f"   Already archived in {len(existing_archives)} directories:")
    for i, archive in enumerate(existing_archives, 1):
        print(f"      {i}. {archive}")
    
    # Get user input
    while True:
        user_input = input("\n❓ Do you want to create new archive (y/n)? ").strip().lower()
        if user_input in ["y", "yes"]:
            return True
        elif user_input in ["n", "no"]:
            print("ℹ️  Operation cancelled by user")
            return False
        else:
            print("❌ Invalid input! Please enter 'y' or 'n'")

def create_archive_directory(model_hash):
    """Create unique archive directory with model hash in name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add first 8 chars of hash to directory name for easy identification
    vis_output_dir = os.path.join(RESULTS_DIR, f"train_vis_{timestamp}_{model_hash[:8]}")
    
    # Create directories
    os.makedirs(vis_output_dir, exist_ok=True)
    os.makedirs(os.path.join(vis_output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(vis_output_dir, "model"), exist_ok=True)
    
    print(f"✅ Archive directory created: {vis_output_dir}")
    return vis_output_dir

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
def plot_train_curves(history_df, vis_output_dir):
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
    plot_path = os.path.join(vis_output_dir, "plots", f"train_curves.{IMG_FORMAT}")
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
def archive_model_file(model_source_path, vis_output_dir):
    """Copy .keras model file to archive directory with validation"""
    model_dest_path = os.path.join(vis_output_dir, "model", "best_tomato_model.keras")
    
    # Check if destination exists
    if os.path.exists(model_dest_path):
        print(f"⚠️  Existing model file in archive: {model_dest_path}")
        while True:
            user_input = input("❓ Overwrite existing model file (y/n)? ").strip().lower()
            if user_input in ["y", "yes"]:
                shutil.copy2(model_source_path, model_dest_path)
                print(f"✅ Model file overwritten: {model_dest_path}")
                break
            elif user_input in ["n", "no"]:
                print(f"ℹ️  Skipping model file copy")
                model_dest_path = None
                break
            else:
                print("❌ Invalid input! Please enter 'y' or 'n'")
    else:
        shutil.copy2(model_source_path, model_dest_path)
        print(f"✅ Model file archived to: {model_dest_path}")
    
    return model_dest_path

# ===================== Utility: Summarize Model Info =====================
def summarize_model_info(model, label_to_idx, key_metrics, model_archive_path, model_hash, vis_output_dir):
    """Summarize core model information with hash for unique identification"""
    # 1. Get model summary as text
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary_text = "\n".join(model_summary)

    # 2. Safe shape conversion (fix NoneType error)
    def convert_shape(shape):
        """Safe conversion of model shape with None handling"""
        # Handle None shape
        if shape is None:
            return None
        # Handle TensorFlow shape object
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()
        # Handle non-iterable shapes
        if not isinstance(shape, (list, tuple)):
            return int(shape) if shape is not None else None
        # Convert each dimension to int (handle None in dimensions)
        return [int(dim) if dim is not None else None for dim in shape]

    # 3. Get safe input/output shapes (handle model with multiple inputs/outputs)
    try:
        input_shape = convert_shape(model.input_shape[0] if isinstance(model.input_shape, (list, tuple)) else model.input_shape)
    except:
        input_shape = None
    
    try:
        output_shape = convert_shape(model.output_shape[0] if isinstance(model.output_shape, (list, tuple)) else model.output_shape)
    except:
        output_shape = None

    # 4. Core model information dictionary
    model_info = {
        "basic_info": {
            "model_name": model.name,
            "model_hash": model_hash,  # Unique model identifier
            "total_parameters": int(model.count_params()),
            "trainable_parameters": int(sum([np.prod(layer.shape) for layer in model.trainable_weights])),
            "non_trainable_parameters": int(sum([np.prod(layer.shape) for layer in model.non_trainable_weights])),
            "input_shape": input_shape,
            "output_shape": output_shape,
            "archive_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_archive_path": model_archive_path,
            "archive_directory": vis_output_dir
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

    # 5. Save model info with custom JSON encoder
    info_path = os.path.join(vis_output_dir, "model_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
    print(f"✅ Model information saved to: {info_path}")

    # 6. Save model summary as text
    summary_path = os.path.join(vis_output_dir, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(model_summary_text)
    print(f"✅ Model summary saved to: {summary_path}")

    return model_info

# ===================== Main Function =====================
def main():
    """Main workflow with model validation and duplicate protection"""
    print("🚀 Starting model training information visualization with validation...")
    
    # 1. Define model path
    model_path = os.path.join(PROCESSED_DIR, "best_tomato_model.keras")
    
    # 2. Model validation and duplicate check
    try:
        # Calculate model hash
        model_hash = calculate_model_hash(model_path)
        
        # Check existing archives
        existing_archives = check_existing_archives(model_hash)
        
        # Get user confirmation
        if not get_user_confirmation(existing_archives, model_path):
            return  # Exit if user cancels
        
        # Create archive directory
        vis_output_dir = create_archive_directory(model_hash)
        
        # 3. Load training data
        history_df, label_to_idx, model = load_train_data(model_path)
        
        # 4. Plot training curves
        key_metrics = plot_train_curves(history_df, vis_output_dir)
        
        # 5. Archive model file
        model_archive_path = archive_model_file(model_path, vis_output_dir)
        
        # 6. Summarize and save model information
        model_info = summarize_model_info(model, label_to_idx, key_metrics, model_archive_path, model_hash, vis_output_dir)
        
        # 7. Print final summary
        print("\n📊 Model Training Core Metrics Summary:")
        print(f"   - Maximum Validation Accuracy: {model_info['training_metrics']['max_val_accuracy']:.4f} (Epoch {model_info['training_metrics']['best_epoch_val_acc']})")
        print(f"   - Minimum Validation Loss: {model_info['training_metrics']['min_val_loss']:.4f}")
        print(f"   - Total Model Parameters: {model_info['basic_info']['total_parameters']:,}")
        print(f"   - Trainable Parameters: {model_info['basic_info']['trainable_parameters']:,}")
        print(f"   - Number of Classes: {model_info['dataset_info']['number_of_classes']}")
        print(f"   - Model Hash: {model_hash}")

        print(f"\n🎉 All results archived to: {vis_output_dir}")
        print(f"   - Training curves: plots/train_curves.{IMG_FORMAT}")
        print(f"   - Model file: model/best_tomato_model.keras")
        print(f"   - Model information: model_info.json")
        print(f"   - Model summary: model_summary.txt")
        
    except Exception as e:
        print(f"\n❌ Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        raise

if __name__ == "__main__":
    main()