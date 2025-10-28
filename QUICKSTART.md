# 🚀 Quick Start Guide

Get started with the Pneumonia Classifier in minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## 🔧 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- torchvision (image processing)
- scikit-learn (evaluation metrics)
- matplotlib (visualization)
- And other required packages

### 2. Download Dataset

**Option A: Kaggle (Recommended)**

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account Settings → API → Create New API Token
3. Download the dataset:
   ```bash
   pip install kaggle
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/
   ```

**Option B: Manual Download**

1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Download and extract to `data/` folder
3. Ensure structure:
   ```
   data/chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── val/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── test/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

## 🎯 Usage

### Method 1: Command Line (Easiest)

#### Train a Model

```bash
python train.py \
  --train_dir data/chest_xray/train \
  --val_dir data/chest_xray/val \
  --test_dir data/chest_xray/test \
  --epochs 10 \
  --batch_size 32 \
  --save_plots
```

**Parameters:**
- `--train_dir`: Path to training data
- `--val_dir`: Path to validation data
- `--test_dir`: Path to test data (optional)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--save_plots`: Save training plots

#### Make Predictions

```bash
python predict.py \
  --model outputs/pneumonia_model.pth \
  --image path/to/chest_xray.jpg \
  --visualize
```

**Parameters:**
- `--model`: Path to trained model
- `--image`: Path to X-ray image
- `--visualize`: Show image with prediction

### Method 2: Python Script

```python
import importlib.util

# Load the classifier
spec = importlib.util.spec_from_file_location(
    "pneumonia_classifier", "pneumenia-classifier.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PneumoniaClassifier = module.PneumoniaClassifier

# Initialize
classifier = PneumoniaClassifier(img_size=224)

# Load data
classifier.load_data(
    train_dir='data/chest_xray/train',
    val_dir='data/chest_xray/val',
    batch_size=32
)

# Train
classifier.train(epochs=10)

# Save
classifier.save_model('my_model.pth')

# Predict
prediction, confidence = classifier.predict('xray.jpg')
print(f"{prediction}: {confidence*100:.2f}%")
```

### Method 3: Google Colab

1. Upload `pneumenia-classifier.py` to Colab
2. Enable GPU: `Runtime → Change runtime type → GPU`
3. Run the cells in the notebook
4. Follow the workflow example at the end of the file

## 📊 Expected Results

After training for 10 epochs, you should see:

- **Training Accuracy**: ~90-92%
- **Validation Accuracy**: ~88-90%
- **Test Accuracy**: ~85-92%

Training time:
- **CPU**: ~2-3 hours
- **GPU (Tesla T4)**: ~15-20 minutes

## 🎓 Next Steps

1. **Improve Accuracy**:
   - Train for more epochs (15-20)
   - Use data augmentation
   - Try transfer learning (ResNet, VGG)

2. **Experiment**:
   - Adjust learning rate
   - Change batch size
   - Modify model architecture

3. **Deploy**:
   - Create a web interface
   - Build a REST API
   - Deploy to cloud platforms

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --batch_size 16
```

### Slow Training
```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Use GPU explicitly
python train.py --device cuda
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📚 More Examples

Check `example_usage.py` for more detailed examples:
- Custom training configurations
- Batch predictions
- Visualization techniques
- Advanced usage patterns

## 💡 Tips

1. **Start Small**: Train for 2-3 epochs first to verify everything works
2. **Monitor Progress**: Watch the validation accuracy to detect overfitting
3. **Save Checkpoints**: Save models after each epoch for best results
4. **Use GPU**: Training is 10-20x faster with GPU acceleration

## 🆘 Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review [example_usage.py](example_usage.py) for code examples
- Open an issue on GitHub for bugs or questions

---

**Ready to start? Run your first training:**

```bash
python train.py \
  --train_dir data/chest_xray/train \
  --val_dir data/chest_xray/val \
  --epochs 5
```

Good luck! 🎉
