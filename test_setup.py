#!/usr/bin/env python3
"""
Test Setup Script
Verifies that all dependencies are installed correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("="*70)
    print("🧪 TESTING SETUP")
    print("="*70)
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib'
    }
    
    failed = []
    
    print("\n📦 Checking packages...\n")
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name:20s} - OK")
        except ImportError as e:
            print(f"❌ {name:20s} - FAILED")
            failed.append(name)
    
    print("\n" + "="*70)
    
    if failed:
        print(f"\n❌ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed correctly!")
        return True

def test_pytorch():
    """Test PyTorch functionality"""
    print("\n" + "="*70)
    print("🔥 TESTING PYTORCH")
    print("="*70)
    
    try:
        import torch
        
        print(f"\n📊 PyTorch version: {torch.__version__}")
        print(f"🐍 Python version: {sys.version.split()[0]}")
        
        # Test CUDA
        cuda_available = torch.cuda.is_available()
        print(f"\n🖥️  CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("   ℹ️  Training will use CPU (slower)")
        
        # Test tensor operations
        print("\n🧮 Testing tensor operations...")
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        print("   ✅ Tensor operations work!")
        
        # Test GPU operations if available
        if cuda_available:
            print("\n🚀 Testing GPU operations...")
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = x_gpu + y_gpu
            print("   ✅ GPU operations work!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PyTorch test failed: {e}")
        return False

def test_classifier():
    """Test if classifier module can be loaded"""
    print("\n" + "="*70)
    print("🏥 TESTING CLASSIFIER MODULE")
    print("="*70)
    
    try:
        import importlib.util
        
        print("\n📂 Loading pneumenia-classifier.py...")
        
        spec = importlib.util.spec_from_file_location(
            "pneumonia_classifier", 
            "pneumenia-classifier.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print("   ✅ Module loaded successfully!")
        
        # Test class instantiation
        print("\n🧠 Testing classifier instantiation...")
        PneumoniaClassifier = module.PneumoniaClassifier
        classifier = PneumoniaClassifier(img_size=224)
        print("   ✅ Classifier created successfully!")
        
        # Check model
        print("\n🔍 Checking model architecture...")
        param_count = sum(p.numel() for p in classifier.model.parameters())
        print(f"   Total parameters: {param_count:,}")
        print("   ✅ Model architecture OK!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Classifier test failed: {e}")
        return False

def test_scripts():
    """Test if training and prediction scripts exist"""
    print("\n" + "="*70)
    print("📝 CHECKING SCRIPTS")
    print("="*70)
    
    import os
    
    scripts = {
        'train.py': 'Training script',
        'predict.py': 'Prediction script',
        'example_usage.py': 'Example usage',
        'pneumenia-classifier.py': 'Main classifier',
        'requirements.txt': 'Dependencies'
    }
    
    print()
    all_exist = True
    
    for script, description in scripts.items():
        if os.path.exists(script):
            print(f"✅ {script:30s} - {description}")
        else:
            print(f"❌ {script:30s} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🏥 PNEUMONIA CLASSIFIER - SETUP TEST")
    print("="*70)
    print("\nThis script will verify your setup is correct.\n")
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("PyTorch", test_pytorch()))
    results.append(("Classifier Module", test_classifier()))
    results.append(("Scripts", test_scripts()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print()
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*70)
    
    if passed == total:
        print(f"\n🎉 SUCCESS! All {total} tests passed!")
        print("\n✅ Your setup is ready!")
        print("\n💡 Next steps:")
        print("   1. Download the dataset (see QUICKSTART.md)")
        print("   2. Run: python train.py --help")
        print("   3. Start training your model!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("\n💡 Please fix the issues above before proceeding.")
        print("   Run: pip install -r requirements.txt")
    
    print("="*70)
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
