"""
Model Evaluation and Analysis Script
Comprehensive evaluation of trained emotion detection model
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from train_emotion_classifier import EmotionDataset
import warnings
warnings.filterwarnings('ignore')


class EmotionModelEvaluator:
    """
    Comprehensive model evaluation and analysis
    """
    
    def __init__(self, model_path='models/best_model'):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to saved model
        """
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n{'='*80}")
        print(f"EMOTION MODEL EVALUATOR")
        print(f"{'='*80}")
        print(f"  Model path: {model_path}")
        print(f"  Device: {self.device}")
        
        # Load model and tokenizer
        print(f"\n✓ Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.num_labels = len(self.id2label)
        
        print(f"  ✓ Model loaded successfully!")
        print(f"  Number of classes: {self.num_labels}")
    
    def load_test_data(self, data_dir='processed_data', batch_size=32, max_length=128):
        """Load test dataset"""
        print(f"\n{'='*80}")
        print(f"LOADING TEST DATA")
        print(f"{'='*80}")
        
        test_df = pd.read_csv(f'{data_dir}/test.csv')
        
        test_dataset = EmotionDataset(
            test_df['cleaned_text'].values,
            test_df['label'].values,
            self.tokenizer,
            max_length
        )
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\n✓ Test data loaded!")
        print(f"  Test samples: {len(test_df):,}")
        print(f"  Test batches: {len(test_loader):,}")
        
        return test_loader, test_df
    
    def evaluate(self, test_loader):
        """Run evaluation on test set"""
        print(f"\n{'='*80}")
        print(f"RUNNING EVALUATION")
        print(f"{'='*80}")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def compute_metrics(self, predictions, labels):
        """Compute comprehensive metrics"""
        print(f"\n{'='*80}")
        print(f"COMPUTING METRICS")
        print(f"{'='*80}")
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        print(f"\n  Overall Metrics:")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support_per_class': support_per_class
        }
    
    def plot_confusion_matrix(self, predictions, labels, save_path='models/confusion_matrix.png'):
        """Plot confusion matrix"""
        print(f"\n✓ Generating confusion matrix...")
        
        cm = confusion_matrix(labels, predictions)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=[self.id2label[i] for i in range(self.num_labels)],
            yticklabels=[self.id2label[i] for i in range(self.num_labels)],
            ax=ax,
            cbar_kws={'label': 'Normalized Frequency'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to: {save_path}")
    
    def plot_per_class_metrics(self, metrics, save_path='models/per_class_metrics.png'):
        """Plot per-class performance"""
        print(f"\n✓ Generating per-class metrics plot...")
        
        emotions = [self.id2label[i] for i in range(self.num_labels)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Emotion': emotions,
            'Precision': metrics['precision_per_class'],
            'Recall': metrics['recall_per_class'],
            'F1-Score': metrics['f1_per_class'],
            'Support': metrics['support_per_class']
        })
        
        # Sort by F1 score
        df = df.sort_values('F1-Score', ascending=False)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # F1 Score
        ax = axes[0, 0]
        ax.barh(df['Emotion'], df['F1-Score'], color='#2E86AB')
        ax.set_xlabel('F1 Score', fontsize=11)
        ax.set_title('F1 Score by Emotion', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Precision
        ax = axes[0, 1]
        ax.barh(df['Emotion'], df['Precision'], color='#A23B72')
        ax.set_xlabel('Precision', fontsize=11)
        ax.set_title('Precision by Emotion', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Recall
        ax = axes[1, 0]
        ax.barh(df['Emotion'], df['Recall'], color='#F18F01')
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_title('Recall by Emotion', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Support
        ax = axes[1, 1]
        ax.barh(df['Emotion'], df['Support'], color='#6A994E')
        ax.set_xlabel('Number of Samples', fontsize=11)
        ax.set_title('Support by Emotion', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to: {save_path}")
    
    def analyze_errors(self, predictions, labels, test_df, save_path='models/error_analysis.txt'):
        """Analyze misclassified examples"""
        print(f"\n✓ Performing error analysis...")
        
        # Find misclassified samples
        misclassified = predictions != labels
        num_errors = misclassified.sum()
        
        error_df = test_df[misclassified].copy()
        error_df['predicted_label'] = predictions[misclassified]
        error_df['true_label'] = labels[misclassified]
        error_df['predicted_emotion'] = error_df['predicted_label'].map(self.id2label)
        error_df['true_emotion'] = error_df['true_label'].map(self.id2label)
        
        # Analyze error patterns
        error_patterns = error_df.groupby(['true_emotion', 'predicted_emotion']).size().reset_index(name='count')
        error_patterns = error_patterns.sort_values('count', ascending=False)
        
        # Save analysis
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ERROR ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total misclassifications: {num_errors:,} / {len(predictions):,} ({num_errors/len(predictions)*100:.2f}%)\n\n")
            
            f.write("Top confusion patterns:\n")
            f.write("-"*80 + "\n")
            for _, row in error_patterns.head(20).iterrows():
                f.write(f"  {row['true_emotion']:15s} → {row['predicted_emotion']:15s}: {row['count']:4d} errors\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("EXAMPLE MISCLASSIFICATIONS\n")
            f.write("="*80 + "\n\n")
            
            # Show examples for top error patterns
            for _, row in error_patterns.head(5).iterrows():
                true_emo = row['true_emotion']
                pred_emo = row['predicted_emotion']
                
                examples = error_df[
                    (error_df['true_emotion'] == true_emo) & 
                    (error_df['predicted_emotion'] == pred_emo)
                ].head(3)
                
                f.write(f"\n{true_emo} → {pred_emo}:\n")
                f.write("-"*80 + "\n")
                for i, (_, ex) in enumerate(examples.iterrows(), 1):
                    f.write(f"{i}. \"{ex['cleaned_text']}\"\n")
                f.write("\n")
        
        print(f"  ✓ Saved to: {save_path}")
        print(f"  Total errors: {num_errors:,} ({num_errors/len(predictions)*100:.2f}%)")
    
    def generate_report(self, metrics, save_path='models/evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        print(f"\n✓ Generating evaluation report...")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EMOTION DETECTION MODEL - EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Number of classes: {self.num_labels}\n\n")
            
            f.write("="*80 + "\n")
            f.write("OVERALL PERFORMANCE\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score:  {metrics['f1']:.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("="*80 + "\n\n")
            
            # Create per-class table
            df = pd.DataFrame({
                'Emotion': [self.id2label[i] for i in range(self.num_labels)],
                'Precision': metrics['precision_per_class'],
                'Recall': metrics['recall_per_class'],
                'F1-Score': metrics['f1_per_class'],
                'Support': metrics['support_per_class'].astype(int)
            })
            
            df = df.sort_values('F1-Score', ascending=False)
            
            f.write(f"{'Emotion':<20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}\n")
            f.write("-"*80 + "\n")
            
            for _, row in df.iterrows():
                f.write(f"{row['Emotion']:<20s} {row['Precision']:>10.4f} {row['Recall']:>10.4f} {row['F1-Score']:>10.4f} {row['Support']:>10d}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("TOP PERFORMING EMOTIONS\n")
            f.write("="*80 + "\n\n")
            
            for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
                f.write(f"{i:2d}. {row['Emotion']:<15s} F1={row['F1-Score']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("LOWEST PERFORMING EMOTIONS\n")
            f.write("="*80 + "\n\n")
            
            for i, (_, row) in enumerate(df.tail(10).iterrows(), 1):
                f.write(f"{i:2d}. {row['Emotion']:<15s} F1={row['F1-Score']:.4f}\n")
        
        print(f"  ✓ Saved to: {save_path}")


def main():
    """Main evaluation pipeline"""
    print("\n" + "="*80)
    print("EMOTION DETECTION MODEL - EVALUATION PIPELINE")
    print("="*80 + "\n")
    
    # Check if model exists
    model_path = 'models/best_model'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first by running: python train_emotion_classifier.py")
        return
    
    # Initialize evaluator
    evaluator = EmotionModelEvaluator(model_path)
    
    # Load test data
    test_loader, test_df = evaluator.load_test_data()
    
    # Run evaluation
    predictions, labels, probabilities = evaluator.evaluate(test_loader)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(predictions, labels)
    
    # Generate visualizations
    evaluator.plot_confusion_matrix(predictions, labels)
    evaluator.plot_per_class_metrics(metrics)
    
    # Error analysis
    evaluator.analyze_errors(predictions, labels, test_df)
    
    # Generate report
    evaluator.generate_report(metrics)
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - models/confusion_matrix.png")
    print("  - models/per_class_metrics.png")
    print("  - models/error_analysis.txt")
    print("  - models/evaluation_report.txt")
    print("\nNext step: Run inference.py to test the model interactively")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
