"""Data Loading and Augmentation Module"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm


class DataLoader:
    """Load and preprocess coffee leaf images"""
    
    def __init__(self, config: dict):
        """Initialize DataLoader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.image_size = tuple(config['data']['image_size'])
        self.n_classes = config['data']['n_classes']
        self.class_names = config['data']['class_names']
        
    def load_images_from_directory(self, directory: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load images from directory structure
        
        Args:
            directory: Path to directory containing class folders
            
        Returns:
            Tuple of (images, labels, filenames)
        """
        images = []
        labels = []
        filenames = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found")
                continue
                
            for filename in tqdm(os.listdir(class_dir), desc=f"Loading {class_name}"):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    try:
                        img = load_img(img_path, target_size=self.image_size)
                        img_array = img_to_array(img)
                        images.append(img_array)
                        labels.append(class_idx)
                        filenames.append(filename)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels), filenames
    
    def create_train_val_test_split(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Create train, validation, test splits
        
        Args:
            X: Images array
            y: Labels array
            
        Returns:
            Dictionary with train, val, test splits
        """
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        test_split = self.config['data']['test_split']
        
        # Train-Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        
        # Train-Val split
        val_size = val_split / (train_split + val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def normalize_images(self, X: np.ndarray) -> np.ndarray:
        """Normalize images to [0, 1] range
        
        Args:
            X: Images array
            
        Returns:
            Normalized images
        """
        return X.astype('float32') / 255.0
    
    def get_augmentation_generator(self):
        """Create ImageDataGenerator with augmentation
        
        Returns:
            ImageDataGenerator instance
        """
        aug_config = self.config['augmentation']
        
        return ImageDataGenerator(
            rotation_range=aug_config['rotation_range'],
            zoom_range=aug_config['zoom_range'],
            width_shift_range=aug_config['width_shift_range'],
            height_shift_range=aug_config['height_shift_range'],
            horizontal_flip=aug_config['horizontal_flip'],
            vertical_flip=aug_config['vertical_flip'],
            fill_mode=aug_config['fill_mode'],
            rescale=1./255
        )
    
    def create_data_generators(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray):
        """Create training and validation data generators
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        train_augmentation = self.get_augmentation_generator()
        val_augmentation = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_augmentation.flow(
            X_train, tf.keras.utils.to_categorical(y_train, self.n_classes),
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        val_generator = val_augmentation.flow(
            X_val, tf.keras.utils.to_categorical(y_val, self.n_classes),
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        return train_generator, val_generator
