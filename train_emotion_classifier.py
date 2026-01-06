"""
Emotion Detection Model Training Pipeline
Fine-tunes transformer models (BERT/RoBERTa/DistilBERT) for emotion classification
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class EmotionClassifierTrainer:
    """
    Comprehensive training pipeline for emotion detection
    """
    
    def __init__(self, model_name='distilbert-base-uncased', device=None):
        """
        Initialize trainer
        
        Args:
            model_name: Pretrained transformer model
            device: cuda or cpu
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.label_mapping = None
        
        print(f"\n{'='*80}")
        print(f"EMOTION CLASSIFIER TRAINER")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_data(self, data_dir='processed_data'):
        """Load preprocessed data"""
        print(f"\n{'='*80}")
        print(f"LOADING DATA")
        print(f"{'='*80}")
        
        # Load datasets
        train_df = pd.read_csv(f'{data_dir}/train.csv')
        val_df = pd.read_csv(f'{data_dir}/val.csv')
        test_df = pd.read_csv(f'{data_dir}/test.csv')
        
        # Load label mapping
        with open(f'{data_dir}/label_mapping.json', 'r') as f:
            self.label_mapping = json.load(f)
        
        self.num_labels = self.label_mapping['num_labels']
        self.id2label = {int(k): v for k, v in self.label_mapping['id2label'].items()}
        self.label2id = self.label_mapping['label2id']
        
        print(f"\n✓ Data loaded successfully!")
        print(f"  Training samples:   {len(train_df):,}")
        print(f"  Validation samples: {len(val_df):,}")
        print(f"  Test samples:       {len(test_df):,}")
        print(f"  Number of classes:  {self.num_labels}")
        
        return train_df, val_df, test_df
    
    def setup_model(self):
        """Initialize model and tokenizer"""
        print(f"\n{'='*80}")
        print(f"SETTING UP MODEL")
        print(f"{'='*80}")
        
        print(f"\n✓ Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"✓ Loading model: {self.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return self
    
    def create_dataloaders(self, train_df, val_df, test_df, batch_size=16, max_length=128):
        """Create PyTorch DataLoaders"""
        print(f"\n{'='*80}")
        print(f"CREATING DATALOADERS")
        print(f"{'='*80}")
        
        print(f"\n  Batch size: {batch_size}")
        print(f"  Max length: {max_length}")
        
        # Create datasets
        train_dataset = EmotionDataset(
            train_df['cleaned_text'].values,
            train_df['label'].values,
            self.tokenizer,
            max_length
        )
        
        val_dataset = EmotionDataset(
            val_df['cleaned_text'].values,
            val_df['label'].values,
            self.tokenizer,
            max_length
        )
        
        test_dataset = EmotionDataset(
            test_df['cleaned_text'].values,
            test_df['label'].values,
            self.tokenizer,
            max_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\n✓ DataLoaders created!")
        print(f"  Training batches:   {len(train_loader):,}")
        print(f"  Validation batches: {len(val_loader):,}")
        print(f"  Test batches:       {len(test_loader):,}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler, epoch, num_epochs):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader, dataset_name="Validation"):
        """Evaluate model"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f'Evaluating {dataset_name}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5, save_dir='models'):
        """Complete training pipeline"""
        print(f"\n{'='*80}")
        print(f"TRAINING")
        print(f"{'='*80}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Optimizer: AdamW")
        print(f"  Scheduler: Linear with warmup")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"  Total steps: {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,}")
        
        # Training loop
        best_val_f1 = 0
        training_stats = []
        
        print(f"\n{'='*80}")
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, scheduler, epoch, epochs
            )
            
            # Validate
            val_metrics = self.evaluate(val_loader, "Validation")
            
            # Print metrics
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
            print(f"  Val F1:     {val_metrics['f1']:.4f} | Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_model(save_dir, epoch, val_metrics)
                print(f"\n  ✓ New best model saved! (F1: {best_val_f1:.4f})")
            
            # Track stats
            training_stats.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall']
            })
        
        print(f"\n{'='*80}")
        print(f"✓ TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"  Best validation F1: {best_val_f1:.4f}")
        
        return training_stats
    
    def save_model(self, save_dir, epoch, metrics):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = f'{save_dir}/best_model'
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save metrics
        with open(f'{save_dir}/best_model_metrics.json', 'w') as f:
            json.dump({
                'epoch': epoch,
                'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                           for k, v in metrics.items() if k not in ['predictions', 'labels']}
            }, f, indent=2)
    
    def evaluate_test_set(self, test_loader, save_dir='models'):
        """Final evaluation on test set"""
        print(f"\n{'='*80}")
        print(f"TEST SET EVALUATION")
        print(f"{'='*80}")
        
        test_metrics = self.evaluate(test_loader, "Test")
        
        print(f"\n{'='*80}")
        print(f"TEST SET RESULTS")
        print(f"{'='*80}")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")
        
        # Detailed classification report
        print(f"\n{'='*80}")
        print(f"DETAILED CLASSIFICATION REPORT")
        print(f"{'='*80}\n")
        
        report = classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=[self.id2label[i] for i in range(self.num_labels)],
            digits=4
        )
        print(report)
        
        # Save test results
        with open(f'{save_dir}/test_results.json', 'w') as f:
            json.dump({
                'accuracy': float(test_metrics['accuracy']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1': float(test_metrics['f1'])
            }, f, indent=2)
        
        # Save classification report
        with open(f'{save_dir}/classification_report.txt', 'w') as f:
            f.write(report)
        
        return test_metrics


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("EMOTION DETECTION MODEL - TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Configuration
    CONFIG = {
        'model_name': 'distilbert-base-uncased',  # Fast and efficient
        'data_dir': 'processed_data',
        'save_dir': 'models',
        'batch_size': 32,
        'max_length': 128,
        'epochs': 3,
        'learning_rate': 2e-5
    }
    
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = EmotionClassifierTrainer(model_name=CONFIG['model_name'])
    
    # Load data
    train_df, val_df, test_df = trainer.load_data(CONFIG['data_dir'])
    
    # Setup model
    trainer.setup_model()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = trainer.create_dataloaders(
        train_df, val_df, test_df,
        batch_size=CONFIG['batch_size'],
        max_length=CONFIG['max_length']
    )
    
    # Train model
    training_stats = trainer.train(
        train_loader, val_loader,
        epochs=CONFIG['epochs'],
        learning_rate=CONFIG['learning_rate'],
        save_dir=CONFIG['save_dir']
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_test_set(test_loader, CONFIG['save_dir'])
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {CONFIG['save_dir']}/best_model/")
    print("\nNext step: Run inference.py to test the model")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
