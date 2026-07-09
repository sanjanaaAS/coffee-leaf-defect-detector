"""Model Evaluation Module"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
)


class Evaluator:
    """Evaluate model performance"""
    
    def __init__(self, config: dict, class_names: list):
        """Initialize Evaluator
        
        Args:
            config: Configuration dictionary
            class_names: List of class names
        """
        self.config = config
        self.class_names = class_names
        
    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model on test set
        
        Args:
            model: Trained model
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray,
                             output_path: str = None):
        """Plot confusion matrix
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            output_path: Path to save plot
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_classification_metrics(self, y_test: np.ndarray, y_pred: np.ndarray,
                                   output_path: str = None):
        """Plot per-class metrics
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            output_path: Path to save plot
        """
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics_dict = {}
        for class_name in self.class_names:
            class_metrics = report.get(class_name, {})
            metrics_dict[class_name] = {
                'precision': class_metrics.get('precision', 0),
                'recall': class_metrics.get('recall', 0),
                'f1-score': class_metrics.get('f1-score', 0)
            }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['precision', 'recall', 'f1-score']
        for idx, metric in enumerate(metrics):
            values = [metrics_dict[cn][metric] for cn in self.class_names]
            axes[idx].bar(self.class_names, values, color='steelblue')
            axes[idx].set_title(f'{metric.capitalize()}')
            axes[idx].set_ylim([0, 1])
            axes[idx].set_ylabel('Score')
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_classification_report(self, y_test: np.ndarray, y_pred: np.ndarray):
        """Print detailed classification report
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
        """
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_test, y_pred, target_names=self.class_names))
