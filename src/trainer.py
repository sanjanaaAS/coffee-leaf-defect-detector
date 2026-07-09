"""Model Training Module"""

import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)


class Trainer:
    """Train coffee leaf disease detection models"""
    
    def __init__(self, config: dict):
        """Initialize Trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.history = None
        
    def get_callbacks(self) -> list:
        """Get training callbacks
        
        Returns:
            List of callbacks
        """
        callbacks = []
        checkpoint_dir = self.config['callbacks']['checkpoint_dir']
        
        # Early Stopping
        if self.config['callbacks']['enable_early_stopping']:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ))
        
        # Reduce Learning Rate
        if self.config['callbacks']['enable_reduce_lr']:
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['training']['reduce_lr_factor'],
                patience=self.config['training']['reduce_lr_patience'],
                verbose=1
            ))
        
        # Model Checkpoint
        if self.config['callbacks']['enable_checkpointing']:
            os.makedirs(checkpoint_dir, exist_ok=True)
            callbacks.append(ModelCheckpoint(
                os.path.join(checkpoint_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ))
        
        # TensorBoard
        log_dir = self.config['callbacks']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        callbacks.append(TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        ))
        
        return callbacks
    
    def train(self, model, train_generator, val_generator, steps_per_epoch=None, 
              validation_steps=None):
        """Train the model
        
        Args:
            model: Keras model
            train_generator: Training data generator
            val_generator: Validation data generator
            steps_per_epoch: Steps per epoch
            validation_steps: Validation steps
            
        Returns:
            Training history
        """
        epochs = self.config['training']['epochs']
        callbacks = self.get_callbacks()
        
        self.history = model.fit(
            train_generator,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1
        )
        
        return self.history
    
    def save_history(self, output_path: str):
        """Save training history to JSON
        
        Args:
            output_path: Path to save history
        """
        if self.history is None:
            print("No training history to save")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        history_dict = {
            'loss': [float(x) for x in self.history.history['loss']],
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
        }
        
        with open(output_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"Training history saved to {output_path}")
