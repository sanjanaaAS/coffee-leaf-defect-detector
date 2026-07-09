# Coffee Leaf Defect Detector - MCA Master Level Project

## Project Overview
An AI-powered system to detect and classify coffee leaf diseases using Deep Learning (CNN) and Transfer Learning. This project implements state-of-the-art computer vision techniques to identify coffee plant defects in real-time.

## Key Features
- ✅ Multi-class disease classification (Healthy, Rust, Blight, Black Rot)
- ✅ Transfer Learning with pre-trained models (ResNet50, MobileNetV2, EfficientNet)
- ✅ Real-time defect detection and localization
- ✅ Web application for inference
- ✅ Mobile deployment support (TensorFlow Lite)
- ✅ Comprehensive model evaluation & visualization
- ✅ Grad-CAM visualization for model explainability

## Project Structure
```
coffee-leaf-defect-detector/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Preprocessed images
│   └── splits/                 # Train/Val/Test splits
├── models/
│   ├── trained_models/         # Saved model weights
│   └── model_configs/          # Model configuration files
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading & augmentation
│   ├── preprocessing.py        # Image preprocessing
│   ├── model_builder.py        # Model architecture
│   ├── trainer.py              # Training loop
│   ├── evaluator.py            # Model evaluation
│   └── visualizer.py           # Visualization & Grad-CAM
├── app/
│   ├── flask_app.py            # Flask web application
│   ├── templates/              # HTML templates
│   └── static/                 # CSS, JS, uploaded files
├── configs/
│   └── config.yaml             # Configuration parameters
├── tests/
│   ├── test_data_loader.py
│   └── test_model.py
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── LICENSE
```

## Technology Stack
- **Language**: Python 3.9+
- **ML Frameworks**: TensorFlow 2.x, Keras
- **Computer Vision**: OpenCV, scikit-image
- **Deep Learning**: ResNet, MobileNetV2, EfficientNet
- **Web Framework**: Flask
- **Visualization**: Matplotlib, Seaborn, Grad-CAM
- **Data Processing**: NumPy, Pandas, Pillow

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sanjanaaAS/coffee-leaf-defect-detector.git
cd coffee-leaf-defect-detector
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset

### Sources
- **Kaggle**: [Coffee Leaf Disease Dataset](https://www.kaggle.com/)
- **PlantVillage**: Plant disease detection dataset
- **Custom Collection**: Smartphone/drone imagery from coffee plantations

### Dataset Classes
1. **Healthy Leaf** - Normal coffee leaves without disease
2. **Coffee Leaf Rust (CLR)** - Orange/brown powder-like appearance
3. **Coffee Berry Disease (CBD)** - Black spots and fruit rot
4. **Black Rot** - Dark lesions on leaves

### Data Preparation
```python
# Download dataset from Kaggle
kaggle datasets download -d <dataset-name>

# Run preprocessing
python src/preprocessing.py --input data/raw --output data/processed
```

## Model Architecture

### Transfer Learning Approach
Using pre-trained ImageNet weights as initialization:

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model

# Load pre-trained model
base_model = MobileNetV2(input_shape=(224, 224, 3), 
                         include_top=False, 
                         weights='imagenet')

# Add custom layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
```

## Training

### Training Script
```bash
python src/trainer.py \\
  --epochs 50 \\
  --batch_size 32 \\
  --learning_rate 0.001 \\
  --model_name mobilenetv2 \\
  --output_dir models/trained_models/
```

### Key Hyperparameters
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Epochs**: 50-100
- **Image Size**: 224x224 pixels
- **Data Augmentation**: Rotation, Zoom, Flip, Brightness adjustment

## Model Evaluation

### Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Disease detection accuracy per class
- **Recall**: Sensitivity to detect each disease
- **F1-Score**: Harmonic mean of precision & recall
- **Confusion Matrix**: Detailed classification results
- **ROC-AUC**: Model discrimination ability

### Evaluation Command
```bash
python src/evaluator.py \\
  --model_path models/trained_models/best_model.h5 \\
  --test_data data/processed/test/ \\
  --output results/
```

## Visualization & Interpretability

### Grad-CAM Visualization
Generates saliency maps showing which leaf regions contributed to predictions:

```bash
python src/visualizer.py \\
  --model_path models/trained_models/best_model.h5 \\
  --image_path data/processed/test/sample.jpg \\
  --output results/gradcam/
```

## Web Application

### Run Flask App
```bash
cd app/
python flask_app.py
```

**Access at**: `http://localhost:5000`

### Features
- Upload coffee leaf image
- Real-time disease prediction
- Confidence score display
- Grad-CAM visualization
- Recommendation suggestions

## Mobile Deployment

### Convert to TensorFlow Lite
```python
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('models/trained_models/best_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('models/mobile_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Expected Results

### Target Performance Metrics
- **Overall Accuracy**: >95%
- **Per-Class Precision**: >93%
- **Per-Class Recall**: >93%
- **F1-Score**: >94%

### Inference Time
- **CPU**: ~200-500ms per image
- **GPU**: ~50-100ms per image
- **Mobile (TFLite)**: ~150-300ms per image

## Project Phases

### Phase 1: Data Preparation (Week 1-2)
- [ ] Download & explore dataset
- [ ] Data augmentation
- [ ] Train/validation/test splits
- [ ] EDA & visualization

### Phase 2: Model Development (Week 3-4)
- [ ] Implement data loader
- [ ] Build transfer learning models
- [ ] Hyperparameter tuning
- [ ] Model training & validation

### Phase 3: Evaluation & Optimization (Week 5-6)
- [ ] Comprehensive evaluation
- [ ] Grad-CAM visualization
- [ ] Error analysis
- [ ] Model optimization (pruning, quantization)

### Phase 4: Deployment (Week 7-8)
- [ ] Build Flask web app
- [ ] Convert to TFLite for mobile
- [ ] Documentation
- [ ] Presentation preparation

## References & Research Papers

1. **Transfer Learning in Plant Disease Detection** - IEEE Conference on CVPR 2023
2. **YOLOv8 for Object Detection** - Ultralytics Research
3. **Vision Transformers for Agricultural AI** - Meta AI Research
4. **Grad-CAM: Visual Explanations** - Selvaraju et al., 2016
5. **PlantVillage Dataset Paper** - Hughes et al., 2016

## Contributing
Fork the repository and create a pull request for improvements.

## License
MIT License - See LICENSE file

## Contact
**Author**: Sanjana AS  
**Email**: sanjana@example.com  
**GitHub**: [@sanjanaaAS](https://github.com/sanjanaaAS)

---

**Last Updated**: July 2026  
**Status**: In Development
