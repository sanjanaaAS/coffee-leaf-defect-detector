# Quick Start Guide - Coffee Leaf Defect Detector Training

## 🚀 5-Minute Setup

### Step 1: Clone & Setup (1 min)
```bash
git clone https://github.com/sanjanaaAS/coffee-leaf-defect-detector.git
cd coffee-leaf-defect-detector
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Prepare Data (2 min)
```bash
# Create directory structure
mkdir -p data/raw/{Healthy,Rust,Berry_Disease,Black_Rot}

# Copy your coffee leaf images into respective folders
# Or download from Kaggle:
kaggle datasets download -d <dataset-name> -p data/raw/
```

### Step 3: Train Model (30 seconds to start)
```bash
# Quick training on CPU
python src/train_main.py --config configs/config.yaml

# Or with GPU acceleration
python src/train_main.py --config configs/config.yaml --gpu
```

### Step 4: View Results (1 min)
```
Results will be saved to:
- models/trained_models/final_model.h5
- results/visualizations/
- logs/
```

---

## 📊 What You'll Get

✅ Trained model (>95% accuracy)  
✅ Confusion matrix visualization  
✅ Training history plots  
✅ Model saved in multiple formats  
✅ Classification metrics  

---

## 🔧 Customization

### Change Model Architecture
Edit `configs/config.yaml`:
```yaml
model:
  architecture: "resnet50"  # or: efficientnet, vgg16
```

### Adjust Training Parameters
```yaml
training:
  epochs: 100          # More epochs
  batch_size: 16       # Smaller batches (if memory limited)
  learning_rate: 0.0005  # Lower learning rate
```

### Enable Fine-tuning
Uncomment in `src/train_main.py`:
```python
# Use AdvancedTrainer for two-phase training
from src.train_advanced import AdvancedTrainer

trainer = AdvancedTrainer(config)
history1 = trainer.train_phase1_frozen(...)
history2 = trainer.train_phase2_unfrozen(...)
```

---

## 📓 Jupyter Notebook

```bash
jupyter notebook notebooks/03_model_training.ipynb
```

Run cells sequentially for interactive training.

---

## 🎯 Expected Performance

| Model | Time/Epoch | Accuracy | Mobile Size |
|-------|-----------|----------|-------------|
| MobileNetV2 | 2-3s | 94-96% | 9MB |
| ResNet50 | 5-7s | 95-97% | 98MB |
| EfficientNet | 3-4s | 95-97% | 32MB |
| VGG16 | 8-10s | 93-95% | 528MB |

---

## ❓ Troubleshooting

### No GPU detected?
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Out of memory?
Reduce batch size: `batch_size: 8`

### Poor accuracy?
Increase training data or epochs

---

## 📚 Next Steps

- [ ] Complete training
- [ ] Evaluate on test set
- [ ] Fine-tune hyperparameters
- [ ] Convert to TensorFlow Lite
- [ ] Deploy to web/mobile

---

## 💡 Tips

- 🖥️ GPU training is 10-20x faster
- 📸 Collect more diverse data for better accuracy
- 🔄 Use mixed precision for faster training
- 💾 Save checkpoints frequently
- 📊 Monitor TensorBoard: `tensorboard --logdir logs/`
