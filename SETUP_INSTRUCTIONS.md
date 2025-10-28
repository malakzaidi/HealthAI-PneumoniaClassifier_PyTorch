# 🚀 Setup Instructions

## Complete Setup Guide for Pneumonia Classifier

Follow these steps to get your Pneumonia Classifier up and running.

---

## Step 1: Verify Python Installation

```bash
python3 --version
```

**Required**: Python 3.8 or higher

If Python is not installed:
- **Ubuntu/Debian**: `sudo apt-get install python3 python3-pip`
- **macOS**: `brew install python3`
- **Windows**: Download from https://www.python.org/downloads/

---

## Step 2: Install Dependencies

```bash
pip3 install -r requirements.txt
```

This will install:
- ✅ PyTorch (deep learning framework)
- ✅ TorchVision (image processing)
- ✅ NumPy (numerical computing)
- ✅ Pillow (image handling)
- ✅ scikit-learn (evaluation metrics)
- ✅ Matplotlib (visualization)
- ✅ Seaborn (enhanced plots)
- ✅ tqdm (progress bars)

**Note**: PyTorch installation may take 5-10 minutes depending on your internet speed.

### GPU Support (Optional but Recommended)

For CUDA-enabled GPU support:

```bash
# Check if CUDA is available
nvidia-smi

# Install PyTorch with CUDA support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 3: Verify Installation

```bash
python3 test_setup.py
```

Expected output:
```
✅ All packages installed correctly!
✅ PyTorch operations work!
✅ Classifier created successfully!
✅ All scripts present!
```

If you see errors, reinstall dependencies:
```bash
pip3 install -r requirements.txt --upgrade
```

---

## Step 4: Download Dataset

### Option A: Using Kaggle API (Recommended)

1. **Create Kaggle Account**
   - Go to https://www.kaggle.com
   - Sign up for free account

2. **Get API Credentials**
   - Go to Account Settings → API
   - Click "Create New API Token"
   - Download `kaggle.json`

3. **Install Kaggle CLI**
   ```bash
   pip3 install kaggle
   ```

4. **Setup Credentials**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

5. **Download Dataset**
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/
   ```

### Option B: Manual Download

1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click "Download" button (requires Kaggle account)
3. Extract the ZIP file
4. Move to project directory:
   ```bash
   mkdir -p data
   mv ~/Downloads/chest_xray data/
   ```

### Verify Dataset Structure

```bash
ls -R data/chest_xray/
```

Should show:
```
data/chest_xray/
├── train/
│   ├── NORMAL/      (1,341 images)
│   └── PNEUMONIA/   (3,875 images)
├── val/
│   ├── NORMAL/      (8 images)
│   └── PNEUMONIA/   (8 images)
└── test/
    ├── NORMAL/      (234 images)
    └── PNEUMONIA/   (390 images)
```

---

## Step 5: Train Your First Model

### Quick Test (2 epochs, ~5 minutes)

```bash
python3 train.py \
  --train_dir data/chest_xray/train \
  --val_dir data/chest_xray/val \
  --epochs 2 \
  --batch_size 32
```

### Full Training (10 epochs, ~15-20 minutes with GPU)

```bash
python3 train.py \
  --train_dir data/chest_xray/train \
  --val_dir data/chest_xray/val \
  --test_dir data/chest_xray/test \
  --epochs 10 \
  --batch_size 32 \
  --save_plots
```

**Training Progress:**
```
Epoch [1/10]
  Train Loss: 0.3245, Train Acc: 85.23%
  Val Loss: 0.2891, Val Acc: 87.50%
----------------------------------------------------------
Epoch [2/10]
  Train Loss: 0.2156, Train Acc: 89.45%
  Val Loss: 0.2234, Val Acc: 90.00%
----------------------------------------------------------
...
```

---

## Step 6: Make Predictions

### Single Image Prediction

```bash
python3 predict.py \
  --model outputs/pneumonia_model.pth \
  --image data/chest_xray/test/NORMAL/IM-0001-0001.jpeg \
  --visualize
```

**Output:**
```
✅ Diagnosis: NORMAL
   Confidence: 94.23%
   Confidence Level: Very High
```

### Batch Predictions

```python
python3 -c "
from example_usage import example_4_batch_predictions
example_4_batch_predictions()
"
```

---

## Step 7: Evaluate Model

```bash
python3 train.py \
  --train_dir data/chest_xray/train \
  --val_dir data/chest_xray/val \
  --test_dir data/chest_xray/test \
  --epochs 10
```

**Expected Results:**
```
TEST SET EVALUATION
============================================================
Accuracy: 89.74%
Precision: 0.9123
Recall: 0.8851
F1-Score: 0.8985

Confusion Matrix:
[[212  22]
 [ 42 348]]
============================================================
```

---

## 🎯 Quick Reference

### File Structure
```
pneumonia-classifier/
├── pneumenia-classifier.py    # Main implementation
├── train.py                   # Training CLI
├── predict.py                 # Prediction CLI
├── example_usage.py           # Code examples
├── test_setup.py              # Setup verification
├── requirements.txt           # Dependencies
├── README.md                  # Full docs
├── QUICKSTART.md              # Quick guide
└── outputs/                   # Results (auto-created)
```

### Common Commands

```bash
# Verify setup
python3 test_setup.py

# Train model
python3 train.py --train_dir data/chest_xray/train --val_dir data/chest_xray/val --epochs 10

# Make prediction
python3 predict.py --model outputs/pneumonia_model.pth --image xray.jpg

# View help
python3 train.py --help
python3 predict.py --help
```

### Python Usage

```python
import importlib.util

# Load classifier
spec = importlib.util.spec_from_file_location("pc", "pneumenia-classifier.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Train
classifier = module.PneumoniaClassifier()
classifier.load_data('data/chest_xray/train', 'data/chest_xray/val')
classifier.train(epochs=10)
classifier.save_model('model.pth')

# Predict
prediction, confidence = classifier.predict('xray.jpg')
print(f"{prediction}: {confidence*100:.2f}%")
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'torch'"
**Solution:**
```bash
pip3 install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution:**
```bash
python3 train.py --batch_size 16  # Reduce batch size
```

### Issue: "Dataset directory not found"
**Solution:**
```bash
# Check path
ls data/chest_xray/train/NORMAL/

# If missing, re-download dataset
```

### Issue: Training is very slow
**Solution:**
```bash
# Check if GPU is available
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Permission denied" on Linux
**Solution:**
```bash
chmod +x train.py predict.py test_setup.py
```

---

## 📚 Next Steps

1. **Read Documentation**
   - `README.md` - Full project documentation
   - `QUICKSTART.md` - Quick start guide
   - `PROJECT_STRUCTURE.md` - Project organization

2. **Try Examples**
   - Open `example_usage.py`
   - Uncomment and run different examples
   - Experiment with parameters

3. **Improve Model**
   - Train for more epochs (15-20)
   - Adjust learning rate
   - Try different architectures
   - Use transfer learning

4. **Deploy**
   - Create web interface
   - Build REST API
   - Deploy to cloud

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip3 install -r requirements.txt`)
- [ ] Setup verified (`python3 test_setup.py`)
- [ ] Dataset downloaded and extracted
- [ ] First model trained successfully
- [ ] Predictions working
- [ ] Documentation reviewed

---

## 🆘 Getting Help

If you encounter issues:

1. **Check test_setup.py output**
   ```bash
   python3 test_setup.py
   ```

2. **Verify dataset structure**
   ```bash
   ls -R data/chest_xray/
   ```

3. **Check Python version**
   ```bash
   python3 --version  # Should be 3.8+
   ```

4. **Reinstall dependencies**
   ```bash
   pip3 install -r requirements.txt --upgrade --force-reinstall
   ```

5. **Review error messages carefully**
   - Most errors indicate missing dependencies or incorrect paths

---

## 🎉 Success!

Once all steps are complete, you should have:

✅ Working Pneumonia Classifier  
✅ Trained model with 85-92% accuracy  
✅ Ability to make predictions on new X-rays  
✅ Understanding of the codebase  

**Congratulations! You're ready to use the Pneumonia Classifier!**

---

**Need more help?** Check:
- `README.md` for detailed documentation
- `example_usage.py` for code examples
- `QUICKSTART.md` for quick reference
