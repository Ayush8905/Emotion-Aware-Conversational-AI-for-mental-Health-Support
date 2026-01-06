"""
Complete Training Pipeline Runner
Executes all steps: preprocessing → training → evaluation → testing
"""

import subprocess
import sys
import os


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text)
    print("="*80 + "\n")


def run_command(command, description):
    """Run a command and handle errors"""
    print_header(description)
    print(f"Running: {command}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully!")


def main():
    """Run complete pipeline"""
    print_header("EMOTION DETECTION - COMPLETE TRAINING PIPELINE")
    
    print("This script will execute:")
    print("  1. Data preprocessing")
    print("  2. Model training")
    print("  3. Model evaluation")
    print("  4. Interactive testing")
    
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Preprocessing
    run_command("python data_preprocessing.py", "STEP 1: Data Preprocessing")
    
    # Step 2: Training
    run_command("python train_emotion_classifier.py", "STEP 2: Model Training")
    
    # Step 3: Evaluation
    run_command("python evaluate_model.py", "STEP 3: Model Evaluation")
    
    # Step 4: Interactive Testing
    print_header("STEP 4: Interactive Testing")
    print("Launching inference mode...")
    print("You can test the model interactively.\n")
    
    subprocess.run("python inference.py", shell=True)
    
    print_header("✓ COMPLETE PIPELINE FINISHED!")
    print("All models and results saved in:")
    print("  - processed_data/")
    print("  - models/")


if __name__ == "__main__":
    main()
