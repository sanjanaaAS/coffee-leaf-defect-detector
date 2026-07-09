"""Visualization and Interpretability Module"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


class Visualizer:
    """Visualize model predictions and create saliency maps"""
    
    def __init__(self, config: dict, class_names: list):
        """Initialize Visualizer
        
        Args:
            config: Configuration dictionary
            class_names: List of class names
        """
        self.config = config
        self.class_names = class_names
        
    def plot_sample_predictions(self, X_test: np.ndarray, y_test: np.ndarray,
                               model, num_samples: int = 9,
                               output_path: str = None):
        """Plot sample predictions
        
        Args:
            X_test: Test images
            y_test: Test labels
            model: Trained model
            num_samples: Number of samples to plot
            output_path: Path to save plot
        """
        predictions = model.predict(X_test[:num_samples])
        predicted_classes = np.argmax(predictions, axis=1)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx in range(num_samples):
            axes[idx].imshow(X_test[idx].astype('uint8'))
            true_label = self.class_names[y_test[idx]]
            pred_label = self.class_names[predicted_classes[idx]]
            confidence = predictions[idx][predicted_classes[idx]]
            
            color = 'green' if y_test[idx] == predicted_classes[idx] else 'red'
            axes[idx].set_title(
                f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                color=color, fontsize=10
            )
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_gradcam(self, model, image: np.ndarray,
                        layer_name: str = None) -> np.ndarray:
        """Generate Grad-CAM visualization
        
        Args:
            model: Trained model
            image: Input image
            layer_name: Name of layer to visualize
            
        Returns:
            Grad-CAM heatmap
        """
        if layer_name is None:
            layer_name = self.config['visualization']['gradcam_layer']
        
        # Get layer
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
        
        # Forward pass
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            class_channel = tf.argmax(predictions[0])
            class_channel = tf.cast(class_channel, tf.int32)
        
        # Compute gradients
        grads = tape.gradient(predictions[:, class_channel], conv_outputs)
        
        # Aggregate gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_gradcam(self, image: np.ndarray, heatmap: np.ndarray,
                       alpha: float = 0.4, output_path: str = None):
        """Overlay Grad-CAM on original image
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            alpha: Transparency
            output_path: Path to save visualization
        """
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(
            image.astype(np.uint8), 1 - alpha,
            heatmap_colored, alpha, 0
        )
        
        return overlay
    
    def plot_gradcam_visualization(self, model, image: np.ndarray,
                                  true_label: str, output_path: str = None):
        """Plot Grad-CAM visualization
        
        Args:
            model: Trained model
            image: Input image
            true_label: True class label
            output_path: Path to save plot
        """
        # Generate Grad-CAM
        image_input = np.expand_dims(image, axis=0)
        heatmap = self.generate_gradcam(model, image_input)
        overlay = self.overlay_gradcam(image, heatmap)
        
        # Get prediction
        prediction = model.predict(image_input)
        pred_class = np.argmax(prediction[0])
        confidence = prediction[0][pred_class]
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image.astype('uint8'))
        axes[0].set_title(f'Original Image\n({true_label})')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(
            f'Prediction: {self.class_names[pred_class]}\nConfidence: {confidence:.2f}'
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()
