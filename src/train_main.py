"""Main Training Script for Coffee Leaf Defect Detector"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import logging

import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy

# Import custom modules
from data_loader import DataLoader
from model_builder import ModelBuilder
from trainer import Trainer
from evaluator import Evaluator
from visualizer import Visualizer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def setup_gpu():
    """Setup GPU for training"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Set GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU setup complete. {len(gpus)} GPU(s) available.")
        except RuntimeError as e:
            logger.error(f"GPU setup failed: {e}")
    else:
        logger.warning("No GPU available. Training will use CPU.")


def prepare_data(config: dict) -> tuple:
    """Prepare and load dataset
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("Preparing data...")
    
    data_loader = DataLoader(config)
    
    # Load images
    raw_data_path = config['data']['raw_data_path']
    logger.info(f"Loading images from {raw_data_path}")
    X, y, filenames = data_loader.load_images_from_directory(raw_data_path)
    
    logger.info(f"Loaded {len(X)} images")
    
    # Create train/val/test splits
    splits = data_loader.create_train_val_test_split(X, y)
    
    # Normalize
    X_train = data_loader.normalize_images(splits['X_train'])
    X_val = data_loader.normalize_images(splits['X_val'])
    X_test = data_loader.normalize_images(splits['X_test'])
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return (
        X_train, splits['y_train'],
        X_val, splits['y_val'],
        X_test, splits['y_test']
    )


def build_and_compile_model(config: dict) -> tf.keras.Model:
    """Build and compile model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Compiled Keras model
    """
    logger.info(f"Building {config['model']['architecture']} model...")
    
    model_builder = ModelBuilder(config)
    model = model_builder.build_transfer_learning_model()
    model = model_builder.compile_model(model)
    
    logger.info("Model compiled successfully")
    logger.info(f"Total parameters: {model.count_params():,}")
    
    return model


def train_model(config: dict, model: tf.keras.Model, 
                X_train, y_train, X_val, y_val):
    """Train the model
    
    Args:
        config: Configuration dictionary
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Training history
    """
    logger.info("Starting training...")
    
    # Create data loaders
    data_loader = DataLoader(config)
    train_generator, val_generator = data_loader.create_data_generators(
        X_train, y_train, X_val, y_val
    )
    
    # Calculate steps per epoch
    steps_per_epoch = max(1, len(X_train) // config['training']['batch_size'])
    validation_steps = max(1, len(X_val) // config['training']['batch_size'])
    
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")
    
    # Train
    trainer = Trainer(config)
    history = trainer.train(
        model, 
        train_generator, 
        val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Save history
    history_path = os.path.join(config['callbacks']['log_dir'], 'training_history.json')
    trainer.save_history(history_path)
    
    logger.info(f"Training complete. History saved to {history_path}")
    
    return history


def plot_training_history(history, output_dir: str):
    """Plot and save training history
    
    Args:
        history: Training history object
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training history plot saved to {output_path}")
    plt.close()


def evaluate_model(config: dict, model: tf.keras.Model, X_test, y_test):
    """Evaluate model on test set
    
    Args:
        config: Configuration dictionary
        model: Trained model
        X_test, y_test: Test data
        
    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    
    evaluator = Evaluator(config, config['data']['class_names'])
    metrics = evaluator.evaluate(model, X_test, y_test)
    
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
    
    # Plot confusion matrix
    output_dir = config['visualization']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator.plot_confusion_matrix(
        metrics['y_test'],
        metrics['y_pred'],
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    evaluator.plot_classification_metrics(
        metrics['y_test'],
        metrics['y_pred'],
        os.path.join(output_dir, 'classification_metrics.png')
    )
    
    evaluator.print_classification_report(metrics['y_test'], metrics['y_pred'])
    
    return metrics


def save_model(model: tf.keras.Model, config: dict):
    """Save trained model
    
    Args:
        model: Trained model
        config: Configuration dictionary
    """
    model_dir = config['callbacks']['checkpoint_dir']
    os.makedirs(model_dir, exist_ok=True)
    
    # Save as H5
    h5_path = os.path.join(model_dir, 'final_model.h5')
    model.save(h5_path)
    logger.info(f"Model saved to {h5_path}")
    
    # Save as SavedModel
    saved_model_path = os.path.join(model_dir, 'final_model')
    model.save(saved_model_path)
    logger.info(f"SavedModel saved to {saved_model_path}")


def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description='Train Coffee Leaf Defect Detector')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()
    
    # Setup GPU if requested
    if args.gpu:
        setup_gpu()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    os.makedirs(config['callbacks']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['callbacks']['log_dir'], exist_ok=True)
    os.makedirs(config['visualization']['output_dir'], exist_ok=True)
    
    try:
        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(config)
        
        # Build model
        model = build_and_compile_model(config)
        model.summary()
        
        # Train model
        history = train_model(config, model, X_train, y_train, X_val, y_val)
        
        # Plot training history
        plot_training_history(history, config['visualization']['output_dir'])
        
        # Evaluate model
        metrics = evaluate_model(config, model, X_test, y_test)
        
        # Save model
        save_model(model, config)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
