# ==============================================================================
# Model Training Information Visualization Script
# File: visualize_train_info_20251210-attention.py
# Path: src/model/visualize_train_info_20251210-attention.py
# Function: Plot training curves + Summarize model info + Archive model + Duplicate protection
# Commit: v2.0.1 (Hotfix) - Fix runtime issues
# ==============================================================================

# ==============================================================================
# Model Training Information Visualization Script
# File: visualize_train_info.py
# Path: src/model/visualize_train_info.py
# Function: Plot training curves + Summarize model info + Archive model + Duplicate protection
# Commit: v2.0.2 (Hotfix) - English UI + Simplified directory naming
# ==============================================================================

import os
import json
import shutil
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    print("⚠️ TensorFlow not found, some features may not work")
    tf = None
from datetime import datetime
from pathlib import Path

# ===================== Core Configuration =====================
# Fix path calculation logic for cross-environment compatibility
try:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
except:
    ROOT_DIR = os.path.abspath(".")

PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Create necessary directories (if not exist)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Image configuration
IMG_FORMAT = "png"
IMG_DPI = 300

# Set matplotlib font
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== Custom JSON Encoder =====================
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if tf and isinstance(obj, tf.TensorShape):
            return obj.as_list()
        return super(NumpyEncoder, self).default(obj)

# ===================== Model Validation & Duplicate Check =====================
def calculate_model_hash(model_path):
    """Calculate MD5 hash for model file unique identification"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    
    # Calculate MD5 hash
    hash_md5 = hashlib.md5()
    try:
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        model_hash = hash_md5.hexdigest()
        print(f"✅ Model MD5 Hash: {model_hash}")
        return model_hash
    except Exception as e:
        print(f"❌ Error calculating model hash: {str(e)}")
        raise

def get_model_archive_dir(model_hash):
    """Get unique archive directory based on model hash (simplified naming)"""
    # Simplified directory name - match example format (no extra suffix)
    return os.path.join(RESULTS_DIR, f"tomato_model_archive_{model_hash}")

def check_model_archive_exists(model_hash):
    """Check if model archive already exists"""
    return os.path.exists(get_model_archive_dir(model_hash))

def get_user_confirmation_for_overwrite(model_hash, model_path):
    """Get user confirmation for overwriting existing archive"""
    archive_dir = get_model_archive_dir(model_hash)
    
    if not os.path.exists(archive_dir):
        return True  # No existing archive, proceed
    
    # Show overwrite warning
    print("\n⚠️ Warning: Model archive already exists!")
    print(f"   Model File: {model_path}")
    print(f"   Existing Archive: {archive_dir}")
    print(f"   Overwriting will replace all files in this directory!")
    
    # Get user input
    try:
        user_input = input("\n❓ Overwrite existing archive (y/n)? ").strip().lower()
        if user_input in ["y", "yes"]:
            # Backup existing files
            backup_dir = f"{archive_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(archive_dir, backup_dir)
            print(f"ℹ️ Existing archive backed up to: {backup_dir}")
            return True
        elif user_input in ["n", "no"]:
            print("ℹ️ Operation cancelled by user")
            return False
        else:
            print("❌ Invalid input! Please enter 'y' or 'n'")
            return get_user_confirmation_for_overwrite(model_hash, model_path)
    except KeyboardInterrupt:
        print("\nℹ️ Operation interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error getting user confirmation: {str(e)}")
        return False

def create_model_archive_dir(model_hash):
    """Create unique archive directory (simplified naming)"""
    archive_dir = get_model_archive_dir(model_hash)
    
    # Create directory
    try:
        os.makedirs(archive_dir, exist_ok=True)
        os.makedirs(os.path.join(archive_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(archive_dir, "model"), exist_ok=True)
        print(f"✅ Archive directory created/updated: {archive_dir}")
        return archive_dir
    except Exception as e:
        print(f"❌ Error creating archive directory: {str(e)}")
        raise

# ===================== Load Training Data =====================
def load_train_data(model_path):
    """Load training history, label mapping and model"""
    # 1. Load training history
    history_paths = [
        os.path.join(PROCESSED_DIR, "train_history.csv"),
        os.path.join(PROCESSED_DIR, "train_history_v1.0.0.csv")
    ]
    
    history_df = None
    for path in history_paths:
        if os.path.exists(path):
            try:
                history_df = pd.read_csv(path)
                print(f"✅ Training history loaded: {history_df.shape} records")
                break
            except Exception as e:
                print(f"⚠️ Error loading {path}: {str(e)}")
    
    if history_df is None:
        raise FileNotFoundError(f"❌ Training history file not found, attempted paths: {history_paths}")

    # 2. Load label mapping
    label_to_idx = None
    label_path = os.path.join(PROCESSED_DIR, "label_to_idx.npy")
    if os.path.exists(label_path):
        try:
            label_to_idx = np.load(label_path, allow_pickle=True).item()
            label_to_idx = {k: int(v) for k, v in label_to_idx.items()}
            print(f"✅ Label mapping loaded: {len(label_to_idx)} classes")
        except Exception as e:
            print(f"⚠️ Error loading label mapping: {str(e)}")
            # Provide default label mapping as fallback
            label_to_idx = {"healthy": 0, "diseased": 1}
            print(f"ℹ️ Using default label mapping: {label_to_idx}")
    else:
        # Provide default label mapping as fallback
        label_to_idx = {"healthy": 0, "diseased": 1}
        print(f"ℹ️ Label mapping file not found, using default values: {label_to_idx}")

    # 3. Load model
    model = None
    if tf and os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print(f"✅ Model loaded: {model.name}")
        except Exception as e:
            print(f"⚠️ Error loading model: {str(e)}")
    else:
        print("⚠️ Model not loaded, some features may be limited")

    return history_df, label_to_idx, model

# ===================== Plot Training Curves (English UI) =====================
def plot_train_curves(history_df, archive_dir):
    """Plot training curves (English labels) and save to archive"""
    try:
        # Create 2x1 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle("Tomato Stress Classification - Training Curves", fontsize=16, fontweight="bold")

        # 1. Plot loss curves (English labels)
        epochs = range(1, len(history_df["loss"]) + 1)
        ax1.plot(epochs, history_df["loss"], "b-", linewidth=2, label="Training Loss")
        if "val_loss" in history_df.columns:
            ax1.plot(epochs, history_df["val_loss"], "r-", linewidth=2, label="Validation Loss")
        
        ax1.set_title("Training & Validation Loss", fontsize=14)
        ax1.set_xlabel("Epochs", fontsize=12)
        ax1.set_ylabel("Loss Value", fontsize=12)
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Annotate minimum loss values
        min_train_loss = history_df["loss"].min()
        ax1.annotate(f"Min: {min_train_loss:.4f}", 
                    xy=(history_df["loss"].idxmin()+1, min_train_loss),
                    xytext=(history_df["loss"].idxmin()+1, min_train_loss+0.1),
                    arrowprops=dict(arrowstyle="->", color="blue"))
        
        if "val_loss" in history_df.columns:
            min_val_loss = history_df["val_loss"].min()
            ax1.annotate(f"Min: {min_val_loss:.4f}", 
                        xy=(history_df["val_loss"].idxmin()+1, min_val_loss),
                        xytext=(history_df["val_loss"].idxmin()+1, min_val_loss+0.1),
                        arrowprops=dict(arrowstyle="->", color="red"))

        # 2. Plot accuracy curves (English labels)
        if "accuracy" in history_df.columns:
            ax2.plot(epochs, history_df["accuracy"], "b-", linewidth=2, label="Training Accuracy")
        if "val_accuracy" in history_df.columns:
            ax2.plot(epochs, history_df["val_accuracy"], "r-", linewidth=2, label="Validation Accuracy")
        
        ax2.set_title("Training & Validation Accuracy", fontsize=14)
        ax2.set_xlabel("Epochs", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.legend(loc="lower right", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Annotate maximum accuracy values
        max_train_acc = history_df["accuracy"].max() if "accuracy" in history_df.columns else 0
        if "accuracy" in history_df.columns:
            ax2.annotate(f"Max: {max_train_acc:.4f}", 
                        xy=(history_df["accuracy"].idxmax()+1, max_train_acc),
                        xytext=(history_df["accuracy"].idxmax()+1, max_train_acc-0.05),
                        arrowprops=dict(arrowstyle="->", color="blue"))
        
        max_val_acc = history_df["val_accuracy"].max() if "val_accuracy" in history_df.columns else 0
        if "val_accuracy" in history_df.columns:
            ax2.annotate(f"Max: {max_val_acc:.4f}", 
                        xy=(history_df["val_accuracy"].idxmax()+1, max_val_acc),
                        xytext=(history_df["val_accuracy"].idxmax()+1, max_val_acc-0.05),
                        arrowprops=dict(arrowstyle="->", color="red"))

        # Save image
        plt.tight_layout()
        plot_path = os.path.join(archive_dir, "plots", f"train_curves.{IMG_FORMAT}")
        plt.savefig(plot_path, dpi=IMG_DPI, bbox_inches="tight")
        plt.close()
        print(f"✅ Training curves saved to: {plot_path}")

        # Return key metrics
        key_metrics = {
            "max_train_accuracy": float(max_train_acc) if "accuracy" in history_df.columns else None,
            "max_val_accuracy": float(max_val_acc) if "val_accuracy" in history_df.columns else None,
            "min_train_loss": float(min_train_loss),
            "min_val_loss": float(min_val_loss) if "val_loss" in history_df.columns else None,
            "best_epoch_val_acc": int(history_df["val_accuracy"].idxmax() + 1) if "val_accuracy" in history_df.columns else None,
            "total_epochs": len(epochs),
            "plot_generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return key_metrics
    except Exception as e:
        print(f"❌ Error plotting training curves: {str(e)}")
        # Return basic metrics to continue execution
        return {
            "error": str(e),
            "total_epochs": len(history_df) if hasattr(history_df, 'shape') else 0,
            "plot_generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ===================== Archive Model File =====================
def archive_model_file(model_source_path, archive_dir):
    """Copy model file to archive directory"""
    try:
        if not os.path.exists(model_source_path):
            print(f"⚠️ Model source file does not exist: {model_source_path}")
            return None
            
        model_dest_path = os.path.join(archive_dir, "model", "best_tomato_model.keras")
        
        # Copy model file
        shutil.copy2(model_source_path, model_dest_path)
        print(f"✅ Model file archived to: {model_dest_path}")
        return model_dest_path
    except Exception as e:
        print(f"⚠️ Error archiving model file: {str(e)}")
        return None

# ===================== Summarize Model Info =====================
def summarize_model_info(model, label_to_idx, key_metrics, model_archive_path, model_hash, archive_dir):
    """Summarize core model information"""
    try:
        # 1. Get model summary text
        model_summary = []
        if model:
            try:
                model.summary(print_fn=lambda x: model_summary.append(x))
                model_summary_text = "\n".join(model_summary)
            except Exception as e:
                model_summary_text = f"Error getting model summary: {str(e)}"
        else:
            model_summary_text = "Model not loaded"

        # 2. Safe shape conversion
        def convert_shape(shape):
            """Safely convert model shape, handle None values"""
            if shape is None:
                return None
            if tf and isinstance(shape, tf.TensorShape):
                shape = shape.as_list()
            if not isinstance(shape, (list, tuple)):
                return int(shape) if shape is not None else None
            return [int(dim) if dim is not None else None for dim in shape]

        # 3. Get input/output shapes
        input_shape = None
        output_shape = None
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        if model:
            try:
                input_shape = convert_shape(model.input_shape[0] if isinstance(model.input_shape, (list, tuple)) else model.input_shape)
            except:
                input_shape = None
            
            try:
                output_shape = convert_shape(model.output_shape[0] if isinstance(model.output_shape, (list, tuple)) else model.output_shape)
            except:
                output_shape = None
                
            try:
                total_params = int(model.count_params())
            except:
                total_params = 0
                
            try:
                trainable_params = int(sum([np.prod(layer.shape) for layer in model.trainable_weights]))
            except:
                trainable_params = 0
                
            try:
                non_trainable_params = int(sum([np.prod(layer.shape) for layer in model.non_trainable_weights]))
            except:
                non_trainable_params = 0

        # 4. Core model info dictionary
        model_info = {
            "basic_info": {
                "model_name": model.name if (model and hasattr(model, 'name')) else "Unknown Model",
                "model_hash": model_hash,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": non_trainable_params,
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

        # 5. Update existing info (if exists)
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
            except Exception as e:
                print(f"⚠️ Error updating existing model info: {str(e)}")

        # 6. Save model info
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
        print(f"✅ Model info saved to: {info_path}")

        # 7. Save model summary
        summary_path = os.path.join(archive_dir, "model_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(model_summary_text)
        print(f"✅ Model summary saved to: {summary_path}")

        return model_info
    except Exception as e:
        print(f"❌ Error summarizing model info: {str(e)}")
        return None

# ===================== Main Function =====================
def main():
    """Main workflow"""
    print("🚀 Starting Model Training Information Visualization (v2.0.2) - Hash-based archiving...")
    
    # 1. Define model path
    model_path = os.path.join(PROCESSED_DIR, "best_tomato_model.keras")
    
    # 2. Core duplicate protection logic
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            # Ask user if they want to continue without model file
            user_input = input("❓ No model file found, continue generating visualization results? (y/n) ").strip().lower()
            if user_input not in ["y", "yes"]:
                print("ℹ️ Operation cancelled")
                return
            # Generate temporary hash for archiving
            model_hash = hashlib.md5(str(datetime.now()).encode()).hexdigest()
            print(f"ℹ️ Generated temporary hash for archiving: {model_hash}")
        else:
            # Calculate model hash
            model_hash = calculate_model_hash(model_path)
        
        # Check archive existence and get user confirmation
        if not get_user_confirmation_for_overwrite(model_hash, model_path):
            return
        
        # Create archive directory
        archive_dir = create_model_archive_dir(model_hash)
        
        # 3. Load training data
        history_df, label_to_idx, model = load_train_data(model_path)
        
        # 4. Plot training curves (English labels)
        key_metrics = plot_train_curves(history_df, archive_dir)
        
        # 5. Archive model file
        model_archive_path = archive_model_file(model_path, archive_dir) if os.path.exists(model_path) else None
        
        # 6. Summarize and save model info
        model_info = summarize_model_info(model, label_to_idx, key_metrics, model_archive_path, model_hash, archive_dir)
        
        # 7. Print final summary
        if model_info:
            print("\n📊 Model Training Core Metrics Summary:")
            if model_info['training_metrics'].get('max_val_accuracy') is not None:
                print(f"   - Max Validation Accuracy: {model_info['training_metrics']['max_val_accuracy']:.4f} (Epoch {model_info['training_metrics']['best_epoch_val_acc']})")
            if model_info['training_metrics'].get('min_val_loss') is not None:
                print(f"   - Min Validation Loss: {model_info['training_metrics']['min_val_loss']:.4f}")
            print(f"   - Total Model Parameters: {model_info['basic_info']['total_parameters']:,}")
            print(f"   - Trainable Parameters: {model_info['basic_info']['trainable_parameters']:,}")
            print(f"   - Number of Classes: {model_info['dataset_info']['number_of_classes']}")
            print(f"   - Model Hash: {model_hash}")
            print(f"   - Archive Directory: {archive_dir}")

        print(f"\n🎉 All results archived to hash-based directory: {archive_dir}")
        print(f"   - Training Curves: plots/train_curves.{IMG_FORMAT}")
        print(f"   - Model File: model/best_tomato_model.keras (if exists)")
        print(f"   - Model Info: model_info.json")
        print(f"   - Model Summary: model_summary.txt")
        
    except Exception as e:
        print(f"\n❌ Error during visualization process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()