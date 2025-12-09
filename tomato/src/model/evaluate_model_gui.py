# ==============================================================================
# Enhanced Model Evaluation GUI Tool
# File: evaluate_model_gui.py
# Path: src/model/evaluate_model_gui.py
# Features: Full metrics + confusion matrix + loss curves + duplicate protection
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

# ===================== Utility: Calculate Model Hash =====================
def calculate_model_hash(model_path):
    """Calculate MD5 hash for model file (unique identifier)"""
    hash_md5 = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

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

# ===================== Utility: Load Validation Data =====================
def load_validation_data():
    """Load validation dataset, label mapping and training history"""
    # Load label mapping
    label_to_idx_path = os.path.join(PROCESSED_DIR, "label_to_idx.npy")
    if not os.path.exists(label_to_idx_path):
        messagebox.showerror("Error", f"Label mapping file not found:\n{label_to_idx_path}")
        return None, None, None, None, None
    
    label_to_idx = np.load(label_to_idx_path, allow_pickle=True).item()
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    
    # Load validation CSV
    val_csv_path = os.path.join(PROCESSED_DIR, "val.csv")
    if not os.path.exists(val_csv_path):
        messagebox.showerror("Error", f"Validation CSV not found:\n{val_csv_path}")
        return None, None, None, None, None
    
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
    """Get list of available models (best_model + results directory models)"""
    model_files = []
    
    # Add best_tomato_model.keras
    best_model_path = os.path.join(PROCESSED_DIR, "best_tomato_model.keras")
    if os.path.exists(best_model_path):
        model_files.append(("Current Best Model", best_model_path))
    
    # Add models from results directory
    if os.path.exists(RESULTS_DIR):
        for root, dirs, files in os.walk(RESULTS_DIR):
            for file in files:
                if file.endswith(".keras") and "best_tomato_model" in file:
                    model_path = os.path.join(root, file)
                    dir_name = os.path.basename(os.path.dirname(model_path))
                    model_files.append((f"Archive: {dir_name}", model_path))
    
    return model_files

# ===================== Utility: Calculate Advanced Metrics =====================
def calculate_classification_metrics(true_labels, pred_labels, num_classes):
    """Calculate precision, recall, F1-score, class accuracy"""
    # Initialize metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    class_accuracy = np.zeros(num_classes)
    
    for cls in range(num_classes):
        # True positive, false positive, false negative
        tp = np.sum((true_labels == cls) & (pred_labels == cls))
        fp = np.sum((true_labels != cls) & (pred_labels == cls))
        fn = np.sum((true_labels == cls) & (pred_labels != cls))
        tn = np.sum((true_labels != cls) & (pred_labels != cls))
        
        # Precision (avoid division by zero)
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall
        recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-score
        f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) > 0 else 0
        
        # Class accuracy
        class_accuracy[cls] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Macro average
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
        # Convert model to concrete function
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_shape = [1] + list(input_shape)[1:]
        
        concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
            tf.TensorSpec(input_shape, model.inputs[0].dtype)
        )
        
        # Convert to frozen graph
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph = frozen_func.graph
        
        # Calculate FLOPs
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        
        return flops.total_float_ops if flops else 0
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
    """Comprehensive model evaluation with all metrics"""
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        num_classes = len(class_names)
        
        # 1. Memory usage measurement
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        gpu_mem_before = tf.config.experimental.get_memory_info('GPU:0')['used'] / 1024 / 1024 if tf.config.list_physical_devices('GPU') else 0
        
        # 2. Inference speed measurement
        # Warm up
        model.predict(val_images[:10], verbose=0)
        
        # Actual inference
        start_time = time.time()
        predictions = model.predict(val_images, verbose=0)
        inference_time = time.time() - start_time
        avg_inference_time = inference_time / len(val_images) * 1000  # ms per image
        fps = len(val_images) / inference_time  # frames per second
        
        # 3. Memory after inference
        mem_after = process.memory_info().rss / 1024 / 1024
        gpu_mem_after = tf.config.experimental.get_memory_info('GPU:0')['used'] / 1024 / 1024 if tf.config.list_physical_devices('GPU') else 0
        mem_usage = mem_after - mem_before
        gpu_mem_usage = gpu_mem_after - gpu_mem_before if tf.config.list_physical_devices('GPU') else 0
        
        # 4. Classification metrics
        pred_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(pred_labels == val_labels) * 100
        
        # Advanced classification metrics
        cls_metrics = calculate_classification_metrics(val_labels, pred_labels, num_classes)
        
        # 5. Model parameters
        total_params = model.count_params()
        trainable_params = sum([np.prod(layer.shape) for layer in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # 6. FLOPs and model size
        flops = calculate_flops(model)
        size_bytes, size_mb, size_gb = calculate_model_size(model_path)
        
        # 7. Confusion matrix
        conf_matrix = tf.math.confusion_matrix(val_labels, pred_labels, num_classes=num_classes).numpy()
        
        return {
            "model": model,
            "predictions": pred_labels,
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

# ===================== Utility: Visualization Functions =====================
def plot_confusion_matrix(canvas, conf_matrix, class_names):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # Clear and display
    for widget in canvas.winfo_children():
        widget.destroy()
    plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
    plot_canvas.draw()
    plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    plt.close(fig)
    return fig

def plot_loss_curves(canvas, train_history):
    """Plot training/validation loss and accuracy curves"""
    if train_history is None:
        return None
    
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
    
    # Clear and display
    for widget in canvas.winfo_children():
        widget.destroy()
    plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
    plot_canvas.draw()
    plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    plt.close(fig)
    return fig

def plot_sample_results(canvas, val_images, val_img_paths, val_labels, pred_labels, idx_to_label):
    """Display sample prediction results"""
    sample_indices = np.random.choice(len(val_images), NUM_SAMPLE_IMAGES, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Model Prediction Results (8 Random Samples)", fontsize=16, fontweight="bold")
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        axes[i].imshow(val_images[idx])
        axes[i].axis("off")
        
        true_label = idx_to_label[val_labels[idx]]
        pred_label = idx_to_label[pred_labels[idx]]
        color = "green" if true_label == pred_label else "red"
        
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", 
                          color=color, fontsize=10)
    
    # Clear and display
    for widget in canvas.winfo_children():
        widget.destroy()
    plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
    plot_canvas.draw()
    plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    plt.close(fig)
    return fig

def plot_class_metrics(canvas, metrics, class_names):
    """Plot class-wise precision, recall, F1-score"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision
    ax1.bar(range(len(class_names)), metrics["precision"], color="skyblue")
    ax1.set_title("Class-wise Precision", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Class", fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha="right")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Recall
    ax2.bar(range(len(class_names)), metrics["recall"], color="lightgreen")
    ax2.set_title("Class-wise Recall", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Class", fontsize=12)
    ax2.set_ylabel("Recall", fontsize=12)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha="right")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # F1-score
    ax3.bar(range(len(class_names)), metrics["f1"], color="salmon")
    ax3.set_title("Class-wise F1-Score", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Class", fontsize=12)
    ax3.set_ylabel("F1-Score", fontsize=12)
    ax3.set_xticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=45, ha="right")
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Clear and display
    for widget in canvas.winfo_children():
        widget.destroy()
    plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
    plot_canvas.draw()
    plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    plt.close(fig)
    return fig

# ===================== Utility: Save Results =====================
def save_evaluation_results(model_name, model_path, eval_results, train_history, class_names, save_dir):
    """Save all evaluation results to files"""
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
    
    # 4. Save sample predictions
    sample_pred_df = pd.DataFrame({
        "true_label": eval_results["predictions"],
        "pred_label": eval_results["predictions"]
    })
    sample_pred_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    
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

# ===================== GUI Main Class =====================
class ModelEvaluationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomato Stress Classification - Advanced Model Evaluation Tool")
        self.root.geometry("1400x900")
        
        # Load validation data
        self.val_images, self.val_labels, self.idx_to_label, self.val_img_paths, self.train_history, self.class_names = load_validation_data()
        if self.val_images is None:
            root.quit()
            return
        
        # Get available models
        self.model_files = get_available_models()
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
        
        # Model selection combo
        ttk.Label(self.select_frame, text="Select Model:").pack(anchor=tk.W, pady=(0, 5))
        self.model_var = tk.StringVar()
        model_names = [name for name, path in self.model_files]
        self.model_combo = ttk.Combobox(self.select_frame, textvariable=self.model_var, values=model_names, state="readonly", width=80)
        self.model_combo.current(0)
        self.model_combo.pack(anchor=tk.W, pady=(0, 10))
        
        # Evaluate button
        self.eval_btn = ttk.Button(self.select_frame, text="Run Comprehensive Evaluation", command=self.run_evaluation)
        self.eval_btn.pack(anchor=tk.W, pady=(0, 20))
        
        # Evaluation results text
        ttk.Label(self.select_frame, text="Evaluation Log:").pack(anchor=tk.W)
        self.results_text = tk.Text(self.select_frame, height=20, width=100)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # 2. Sample Predictions Tab
        self.sample_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.sample_frame, text="Sample Predictions")
        self.sample_canvas = ttk.Frame(self.sample_frame)
        self.sample_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 3. Confusion Matrix Tab
        self.conf_matrix_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.conf_matrix_frame, text="Confusion Matrix")
        self.conf_matrix_canvas = ttk.Frame(self.conf_matrix_frame)
        self.conf_matrix_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 4. Loss Curves Tab
        self.loss_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.loss_frame, text="Training/Validation Curves")
        self.loss_canvas = ttk.Frame(self.loss_frame)
        self.loss_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 5. Class-wise Metrics Tab
        self.class_metrics_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.class_metrics_frame, text="Class-wise Metrics")
        self.class_metrics_canvas = ttk.Frame(self.class_metrics_frame)
        self.class_metrics_canvas.pack(fill=tk.BOTH, expand=True)
    
    def run_evaluation(self):
        """Run full model evaluation workflow"""
        # Get selected model
        selected_idx = self.model_combo.current()
        model_name, model_path = self.model_files[selected_idx]
        
        # Check for duplicate evaluation
        model_hash = calculate_model_hash(model_path)
        is_duplicate, save_dir = check_duplicate_evaluation(model_hash)
        
        if is_duplicate and not get_overwrite_confirmation(model_name):
            self.results_text.insert(tk.END, "Evaluation cancelled by user.\n")
            return
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Starting evaluation for: {model_name}\n")
        self.results_text.insert(tk.END, f"Model path: {model_path}\n")
        self.results_text.insert(tk.END, f"Model hash: {model_hash}\n")
        self.results_text.insert(tk.END, "Please wait - this may take several minutes...\n")
        self.root.update_idletasks()
        
        # Run evaluation
        eval_results = evaluate_model(model_path, self.val_images, self.val_labels, self.class_names)
        if eval_results is None:
            return
        
        # Generate visualizations
        self.results_text.insert(tk.END, "Generating visualizations...\n")
        self.root.update_idletasks()
        
        # Plot sample results
        plot_sample_results(self.sample_canvas, self.val_images, self.val_img_paths,
                           self.val_labels, eval_results["predictions"], self.idx_to_label)
        
        # Plot confusion matrix
        plot_confusion_matrix(self.conf_matrix_canvas, eval_results["conf_matrix"], self.class_names)
        
        # Plot loss curves
        plot_loss_curves(self.loss_canvas, self.train_history)
        
        # Plot class-wise metrics
        plot_class_metrics(self.class_metrics_canvas, eval_results, self.class_names)
        
        # Generate detailed results text
        results_text = f"""
========================================
COMPREHENSIVE MODEL EVALUATION RESULTS
========================================
Model Information:
  - Model Name: {model_name}
  - Model Path: {model_path}
  - Model Hash: {model_hash}
  - Evaluation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

----------------------------------------
CORE CLASSIFICATION METRICS:
  - Overall Accuracy: {eval_results['accuracy']:.2f}%
  - Macro Precision: {eval_results['macro_precision']:.2f}%
  - Macro Recall: {eval_results['macro_recall']:.2f}%
  - Macro F1-Score: {eval_results['macro_f1']:.2f}%

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
  - FLOPs: {eval_results['flops']:,} operations
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
        
        # Save all results
        save_path = save_evaluation_results(
            model_name, model_path, eval_results, 
            self.train_history, self.class_names, save_dir
        )
        
        self.results_text.insert(tk.END, f"\n✅ All results successfully saved to:\n{save_path}")
        messagebox.showinfo("Success", "Evaluation completed successfully!\nAll results have been saved.")

# ===================== Main Function =====================
def main():
    root = tk.Tk()
    app = ModelEvaluationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()