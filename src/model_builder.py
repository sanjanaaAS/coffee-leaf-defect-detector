"""Model Architecture Builder Module"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, EfficientNetB0, VGG16
)


class ModelBuilder:
    """Build deep learning models for coffee leaf classification"""
    
    def __init__(self, config: dict):
        """Initialize ModelBuilder
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.image_size = tuple(config['data']['image_size'])
        self.n_classes = config['data']['n_classes']
        
    def build_transfer_learning_model(self) -> Model:
        """Build transfer learning model
        
        Returns:
            Keras Model
        """
        model_name = self.config['model']['architecture'].lower()
        
        if model_name == 'mobilenetv2':
            return self._build_mobilenetv2()
        elif model_name == 'resnet50':
            return self._build_resnet50()
        elif model_name == 'efficientnet':
            return self._build_efficientnet()
        elif model_name == 'vgg16':
            return self._build_vgg16()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _build_mobilenetv2(self) -> Model:
        """Build MobileNetV2 based model"""
        base_model = MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        if self.config['model']['freeze_backbone']:
            base_model.trainable = False
        
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['model']['dropout_rate'])(x)
        output = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def _build_resnet50(self) -> Model:
        """Build ResNet50 based model"""
        base_model = ResNet50(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        if self.config['model']['freeze_backbone']:
            base_model.trainable = False
        
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['model']['dropout_rate'])(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def _build_efficientnet(self) -> Model:
        """Build EfficientNet based model"""
        base_model = EfficientNetB0(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        if self.config['model']['freeze_backbone']:
            base_model.trainable = False
        
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['model']['dropout_rate'])(x)
        output = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def _build_vgg16(self) -> Model:
        """Build VGG16 based model"""
        base_model = VGG16(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        if self.config['model']['freeze_backbone']:
            base_model.trainable = False
        
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.Dropout(self.config['model']['dropout_rate'])(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def compile_model(self, model: Model) -> Model:
        """Compile model with optimizer and loss
        
        Args:
            model: Keras Model
            
        Returns:
            Compiled model
        """
        optimizer_name = self.config['training']['optimizer'].lower()
        learning_rate = self.config['training']['learning_rate']
        
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=self.config['training']['loss_function'],
            metrics=self.config['training']['metrics']
        )
        
        return model
