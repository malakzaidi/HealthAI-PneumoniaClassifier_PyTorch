#!/usr/bin/env python3
"""
Prediction Script for Pneumonia Classifier
Usage: python predict.py --model pneumonia_model.pth --image chest_xray.jpg
"""

import argparse
import os
import sys
import torch
from PIL import Image
import importlib.util

def load_classifier_module():
    """Load the classifier module dynamically"""
    spec = importlib.util.spec_from_file_location(
        "pneumonia_classifier", 
        "pneumenia-classifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Predict Pneumonia from Chest X-Ray',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file (.pth)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to chest X-ray image')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (must match training)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--visualize', action='store_true',
                        help='Display the image with prediction')
    
    return parser.parse_args()

def validate_inputs(args):
    """Validate input files"""
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Check if image is valid
    try:
        img = Image.open(args.image)
        img.verify()
    except Exception as e:
        print(f"❌ Error: Invalid image file: {e}")
        sys.exit(1)

def visualize_prediction(image_path, prediction, confidence):
    """Display image with prediction"""
    try:
        import matplotlib.pyplot as plt
        
        img = Image.open(image_path)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
        plt.axis('off')
        
        # Color based on prediction
        color = 'red' if prediction == 'PNEUMONIA' else 'green'
        
        title = f'Prediction: {prediction}\nConfidence: {confidence*100:.2f}%'
        plt.title(title, fontsize=16, fontweight='bold', color=color, pad=20)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("⚠️  Matplotlib not available for visualization")
    except Exception as e:
        print(f"⚠️  Error during visualization: {e}")

def main():
    """Main prediction function"""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("=" * 70)
    print("🏥 PNEUMONIA CLASSIFIER - PREDICTION")
    print("=" * 70)
    
    # Validate inputs
    validate_inputs(args)
    
    # Load classifier module
    print("\n📦 Loading classifier module...")
    try:
        classifier_module = load_classifier_module()
        PneumoniaClassifier = classifier_module.PneumoniaClassifier
    except Exception as e:
        print(f"❌ Error loading classifier: {e}")
        sys.exit(1)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  Device: {device}")
    
    # Initialize classifier
    print(f"\n🧠 Initializing model...")
    classifier = PneumoniaClassifier(img_size=args.img_size, device=device)
    
    # Load trained model
    print(f"📂 Loading model: {args.model}")
    try:
        classifier.load_model(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Make prediction
    print(f"\n🔍 Analyzing image: {args.image}")
    print("   Processing...")
    
    try:
        prediction, confidence = classifier.predict(args.image)
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        sys.exit(1)
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 PREDICTION RESULTS")
    print("=" * 70)
    
    # Color output based on prediction
    if prediction == 'PNEUMONIA':
        status_icon = "⚠️ "
        status_color = "\033[91m"  # Red
    else:
        status_icon = "✅"
        status_color = "\033[92m"  # Green
    
    reset_color = "\033[0m"
    
    print(f"\n{status_icon} Diagnosis: {status_color}{prediction}{reset_color}")
    print(f"   Confidence: {confidence*100:.2f}%")
    
    # Confidence interpretation
    if confidence >= 0.9:
        conf_level = "Very High"
    elif confidence >= 0.75:
        conf_level = "High"
    elif confidence >= 0.6:
        conf_level = "Moderate"
    else:
        conf_level = "Low"
    
    print(f"   Confidence Level: {conf_level}")
    
    # Additional information
    print("\n" + "-" * 70)
    print("ℹ️  IMPORTANT NOTES:")
    print("-" * 70)
    print("• This is an AI-assisted diagnostic tool")
    print("• Results should be verified by qualified medical professionals")
    print("• Not intended as a substitute for professional medical advice")
    print("• For research and educational purposes only")
    print("=" * 70)
    
    # Visualize if requested
    if args.visualize:
        print("\n📊 Displaying visualization...")
        visualize_prediction(args.image, prediction, confidence)
    
    # Return prediction for scripting
    return prediction, confidence

if __name__ == '__main__':
    prediction, confidence = main()
    
    # Exit code: 0 for NORMAL, 1 for PNEUMONIA
    sys.exit(0 if prediction == 'NORMAL' else 1)
