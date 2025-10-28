#!/usr/bin/env python3
"""
Example Usage Script for Pneumonia Classifier
This script demonstrates how to use the classifier in your own code
"""

import importlib.util
import torch

def load_classifier_module():
    """Load the classifier module"""
    spec = importlib.util.spec_from_file_location(
        "pneumonia_classifier", 
        "pneumenia-classifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def example_1_basic_training():
    """Example 1: Basic Training"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Training")
    print("="*70)
    
    # Load module
    classifier_module = load_classifier_module()
    PneumoniaClassifier = classifier_module.PneumoniaClassifier
    
    # Initialize classifier
    classifier = PneumoniaClassifier(img_size=224)
    
    # Load data (adjust paths to your dataset)
    print("\n📂 Loading data...")
    classifier.load_data(
        train_dir='data/chest_xray/train',
        val_dir='data/chest_xray/val',
        batch_size=32
    )
    
    # Train model
    print("\n🚀 Training model...")
    classifier.train(epochs=5)
    
    # Save model
    print("\n💾 Saving model...")
    classifier.save_model('my_pneumonia_model.pth')
    
    # Plot training history
    print("\n📊 Plotting training history...")
    classifier.plot_training_history()
    
    print("\n✅ Training complete!")

def example_2_evaluation():
    """Example 2: Model Evaluation"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Model Evaluation")
    print("="*70)
    
    # Load module
    classifier_module = load_classifier_module()
    PneumoniaClassifier = classifier_module.PneumoniaClassifier
    
    # Initialize and load trained model
    classifier = PneumoniaClassifier(img_size=224)
    
    print("\n📂 Loading trained model...")
    classifier.load_model('my_pneumonia_model.pth')
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    accuracy, precision, recall, f1, cm = classifier.evaluate('data/chest_xray/test')
    
    print(f"\n📊 Results:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    print("\n✅ Evaluation complete!")

def example_3_single_prediction():
    """Example 3: Single Image Prediction"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Single Image Prediction")
    print("="*70)
    
    # Load module
    classifier_module = load_classifier_module()
    PneumoniaClassifier = classifier_module.PneumoniaClassifier
    
    # Initialize and load trained model
    classifier = PneumoniaClassifier(img_size=224)
    
    print("\n📂 Loading trained model...")
    classifier.load_model('my_pneumonia_model.pth')
    
    # Predict on a single image
    image_path = 'path/to/chest_xray.jpg'
    
    print(f"\n🔍 Analyzing: {image_path}")
    prediction, confidence = classifier.predict(image_path)
    
    print(f"\n📊 Results:")
    print(f"   Diagnosis: {prediction}")
    print(f"   Confidence: {confidence*100:.2f}%")
    
    print("\n✅ Prediction complete!")

def example_4_batch_predictions():
    """Example 4: Batch Predictions"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Predictions")
    print("="*70)
    
    import os
    
    # Load module
    classifier_module = load_classifier_module()
    PneumoniaClassifier = classifier_module.PneumoniaClassifier
    
    # Initialize and load trained model
    classifier = PneumoniaClassifier(img_size=224)
    
    print("\n📂 Loading trained model...")
    classifier.load_model('my_pneumonia_model.pth')
    
    # Directory with images to predict
    image_dir = 'path/to/images'
    
    print(f"\n🔍 Processing images from: {image_dir}")
    
    results = []
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            prediction, confidence = classifier.predict(image_path)
            results.append({
                'filename': filename,
                'prediction': prediction,
                'confidence': confidence
            })
            print(f"   {filename}: {prediction} ({confidence*100:.2f}%)")
    
    print(f"\n✅ Processed {len(results)} images!")

def example_5_custom_training():
    """Example 5: Custom Training with Parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Training Configuration")
    print("="*70)
    
    # Load module
    classifier_module = load_classifier_module()
    PneumoniaClassifier = classifier_module.PneumoniaClassifier
    
    # Initialize with custom parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = PneumoniaClassifier(img_size=256, device=device)
    
    # Modify learning rate
    for param_group in classifier.optimizer.param_groups:
        param_group['lr'] = 0.0001
    
    print(f"\n🖥️  Device: {device}")
    print(f"   Image size: 256x256")
    print(f"   Learning rate: 0.0001")
    
    # Load data with custom batch size
    print("\n📂 Loading data...")
    classifier.load_data(
        train_dir='data/chest_xray/train',
        val_dir='data/chest_xray/val',
        batch_size=64  # Larger batch size
    )
    
    # Train with more epochs
    print("\n🚀 Training model...")
    classifier.train(epochs=20)
    
    # Save model
    print("\n💾 Saving model...")
    classifier.save_model('custom_pneumonia_model.pth')
    
    print("\n✅ Custom training complete!")

def example_6_visualization():
    """Example 6: Visualize Predictions"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Visualize Predictions")
    print("="*70)
    
    # Load module
    classifier_module = load_classifier_module()
    PneumoniaClassifier = classifier_module.PneumoniaClassifier
    
    # Initialize and load trained model
    classifier = PneumoniaClassifier(img_size=224)
    
    print("\n📂 Loading trained model...")
    classifier.load_model('my_pneumonia_model.pth')
    
    # Visualize predictions on test set
    print("\n📊 Visualizing predictions...")
    classifier.visualize_predictions('data/chest_xray/test', num_images=8)
    
    print("\n✅ Visualization complete!")

def main():
    """Main function to run examples"""
    print("="*70)
    print("🏥 PNEUMONIA CLASSIFIER - EXAMPLE USAGE")
    print("="*70)
    print("\nThis script demonstrates various ways to use the classifier.")
    print("\nAvailable examples:")
    print("  1. Basic Training")
    print("  2. Model Evaluation")
    print("  3. Single Image Prediction")
    print("  4. Batch Predictions")
    print("  5. Custom Training Configuration")
    print("  6. Visualize Predictions")
    
    print("\n" + "="*70)
    print("⚠️  NOTE: Update the data paths in each example before running!")
    print("="*70)
    
    # Uncomment the example you want to run:
    
    # example_1_basic_training()
    # example_2_evaluation()
    # example_3_single_prediction()
    # example_4_batch_predictions()
    # example_5_custom_training()
    # example_6_visualization()
    
    print("\n💡 To run an example, uncomment the corresponding line in main()")
    print("   and update the data paths to match your setup.")

if __name__ == '__main__':
    main()
