#!/usr/bin/env python3
"""
Training Script for Pneumonia Classifier
Usage: python train.py --train_dir data/train --val_dir data/val --epochs 10
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Import the classifier (assuming pneumenia-classifier.py is in the same directory)
# We'll need to handle the import properly
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
        description='Train Pneumonia Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation data directory')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Path to test data directory (optional)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    
    # Model arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save model and results')
    parser.add_argument('--model_name', type=str, default='pneumonia_model.pth',
                        help='Name of the saved model file')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save training plots')
    
    return parser.parse_args()

def validate_directories(args):
    """Validate that data directories exist"""
    if not os.path.exists(args.train_dir):
        print(f"❌ Error: Training directory not found: {args.train_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.val_dir):
        print(f"❌ Error: Validation directory not found: {args.val_dir}")
        sys.exit(1)
    
    if args.test_dir and not os.path.exists(args.test_dir):
        print(f"⚠️  Warning: Test directory not found: {args.test_dir}")
        args.test_dir = None
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✅ Output directory: {args.output_dir}")

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("=" * 70)
    print("🏥 PNEUMONIA CLASSIFIER - TRAINING")
    print("=" * 70)
    
    # Validate directories
    validate_directories(args)
    
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
    
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize classifier
    print(f"\n🧠 Initializing model...")
    print(f"   Image size: {args.img_size}x{args.img_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    
    classifier = PneumoniaClassifier(img_size=args.img_size, device=device)
    
    # Update learning rate if specified
    if args.learning_rate != 0.001:
        for param_group in classifier.optimizer.param_groups:
            param_group['lr'] = args.learning_rate
    
    # Load data
    print(f"\n📂 Loading datasets...")
    print(f"   Train: {args.train_dir}")
    print(f"   Val: {args.val_dir}")
    
    try:
        classifier.load_data(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Train model
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 70)
    
    try:
        classifier.train(epochs=args.epochs)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)
    
    # Save model
    model_path = os.path.join(args.output_dir, args.model_name)
    print(f"\n💾 Saving model to: {model_path}")
    classifier.save_model(model_path)
    
    # Save training plots
    if args.save_plots:
        print(f"\n📊 Saving training plots...")
        import matplotlib.pyplot as plt
        
        classifier.plot_training_history()
        plot_path = os.path.join(args.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {plot_path}")
        plt.close()
    
    # Evaluate on test set if provided
    if args.test_dir:
        print(f"\n🧪 Evaluating on test set...")
        print(f"   Test: {args.test_dir}")
        
        try:
            accuracy, precision, recall, f1, cm = classifier.evaluate(args.test_dir)
            
            # Save evaluation results
            results_path = os.path.join(args.output_dir, 'test_results.txt')
            with open(results_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("TEST SET EVALUATION RESULTS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Accuracy: {accuracy*100:.2f}%\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
                f.write(f"\nConfusion Matrix:\n{cm}\n")
                f.write("=" * 60 + "\n")
            
            print(f"   Results saved to: {results_path}")
            
        except Exception as e:
            print(f"⚠️  Error during evaluation: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Model saved: {model_path}")
    print(f"📁 Output directory: {args.output_dir}")
    
    if len(classifier.val_accs) > 0:
        best_val_acc = max(classifier.val_accs)
        best_epoch = classifier.val_accs.index(best_val_acc) + 1
        print(f"\n🏆 Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    print("\n💡 To use this model for predictions, run:")
    print(f"   python predict.py --model {model_path} --image <path_to_xray.jpg>")
    print("=" * 70)

if __name__ == '__main__':
    main()
