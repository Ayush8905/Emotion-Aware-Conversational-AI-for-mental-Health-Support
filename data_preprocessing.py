"""
Data Preprocessing Module for GoEmotions Dataset
Handles dataset loading, cleaning, analysis, and preparation
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


class EmotionDataPreprocessor:
    """
    Comprehensive data preprocessing for emotion detection
    """
    
    def __init__(self, data_path, model_name='distilbert-base-uncased'):
        """
        Initialize preprocessor
        
        Args:
            data_path: Path to CSV file
            model_name: Transformer model for tokenization
        """
        self.data_path = data_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.emotion_labels = None
        self.df = None
        
    def load_data(self):
        """Load and perform initial data analysis"""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        
        # Try multiple encodings with error handling
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(
                    self.data_path, 
                    encoding=encoding,
                    on_bad_lines='skip',  # Skip problematic lines
                    engine='python'  # More flexible parser
                )
                print(f"\n✓ Dataset loaded successfully with {encoding} encoding!")
                break
            except Exception as e:
                continue
        else:
            raise ValueError("Could not decode CSV file with any standard encoding")
        
        print(f"  Total samples: {len(self.df):,}")
        print(f"  Columns: {len(self.df.columns)}")
        
        # Identify emotion columns
        exclude_cols = ['id', 'text', 'example_very_unclear']
        self.emotion_labels = [col for col in self.df.columns if col not in exclude_cols]
        print(f"  Emotion labels: {len(self.emotion_labels)}")
        
        return self
    
    def analyze_dataset(self):
        """Perform comprehensive dataset analysis"""
        print("\n" + "=" * 80)
        print("DATASET ANALYSIS")
        print("=" * 80)
        
        # Basic statistics
        print("\n1. BASIC STATISTICS:")
        print(f"   - Total samples: {len(self.df):,}")
        print(f"   - Missing values: {self.df['text'].isnull().sum()}")
        
        # Convert boolean columns properly
        if 'example_very_unclear' in self.df.columns:
            self.df['example_very_unclear'] = self.df['example_very_unclear'].map({'False': False, 'True': True, False: False, True: True}).fillna(False)
            print(f"   - Unclear examples: {self.df['example_very_unclear'].sum():,}")
        else:
            print(f"   - Unclear examples: N/A")
        
        # Text length statistics
        self.df['text_length'] = self.df['text'].str.len()
        self.df['word_count'] = self.df['text'].str.split().str.len()
        
        print(f"\n2. TEXT LENGTH STATISTICS:")
        print(f"   - Mean character length: {self.df['text_length'].mean():.1f}")
        print(f"   - Median character length: {self.df['text_length'].median():.1f}")
        print(f"   - Mean word count: {self.df['word_count'].mean():.1f}")
        print(f"   - Max character length: {self.df['text_length'].max()}")
        
        # Emotion distribution
        print(f"\n3. EMOTION DISTRIBUTION:")
        emotion_counts = self.df[self.emotion_labels].sum().sort_values(ascending=False)
        total_emotion_tags = emotion_counts.sum()
        
        print(f"   Total emotion tags: {total_emotion_tags:,}")
        print(f"\n   Top 10 emotions:")
        for i, (emotion, count) in enumerate(emotion_counts.head(10).items(), 1):
            percentage = (count / len(self.df)) * 100
            print(f"   {i:2d}. {emotion:15s}: {count:6,} ({percentage:5.2f}%)")
        
        # Multi-label analysis
        emotion_count_per_sample = self.df[self.emotion_labels].sum(axis=1)
        print(f"\n4. MULTI-LABEL ANALYSIS:")
        print(f"   - Samples with 0 emotions: {(emotion_count_per_sample == 0).sum():,}")
        print(f"   - Samples with 1 emotion:  {(emotion_count_per_sample == 1).sum():,}")
        print(f"   - Samples with 2+ emotions: {(emotion_count_per_sample >= 2).sum():,}")
        print(f"   - Average emotions per sample: {emotion_count_per_sample.mean():.2f}")
        print(f"   - Max emotions in one sample: {emotion_count_per_sample.max()}")
        
        return self
    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Keep basic punctuation for emotional context
        # Don't remove exclamation marks, question marks, etc.
        
        return text
    
    def preprocess_data(self, remove_unclear=True, remove_no_emotion=True):
        """
        Preprocess the dataset
        
        Args:
            remove_unclear: Remove unclear examples
            remove_no_emotion: Remove samples with no emotions
        """
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)
        
        initial_count = len(self.df)
        
        # Remove unclear examples
        if remove_unclear:
            unclear_count = self.df['example_very_unclear'].sum()
            self.df = self.df[self.df['example_very_unclear'] == False].copy()
            print(f"\n✓ Removed {unclear_count:,} unclear examples")
        
        # Clean text
        print(f"✓ Cleaning text...")
        self.df['cleaned_text'] = self.df['text'].apply(self.clean_text)
        
        # Remove empty texts
        empty_count = (self.df['cleaned_text'].str.len() == 0).sum()
        self.df = self.df[self.df['cleaned_text'].str.len() > 0].copy()
        if empty_count > 0:
            print(f"✓ Removed {empty_count:,} empty texts")
        
        # Remove samples with no emotions
        if remove_no_emotion:
            emotion_sum = self.df[self.emotion_labels].sum(axis=1)
            no_emotion_count = (emotion_sum == 0).sum()
            self.df = self.df[emotion_sum > 0].copy()
            if no_emotion_count > 0:
                print(f"✓ Removed {no_emotion_count:,} samples with no emotions")
        
        final_count = len(self.df)
        print(f"\n✓ Preprocessing complete!")
        print(f"  Samples: {initial_count:,} → {final_count:,} (removed {initial_count - final_count:,})")
        
        return self
    
    def create_single_label_dataset(self):
        """
        Convert multi-label to single-label (for classification)
        Takes the first emotion label for each sample
        """
        print("\n" + "=" * 80)
        print("CREATING SINGLE-LABEL DATASET")
        print("=" * 80)
        
        # Get the primary emotion (first one marked)
        def get_primary_emotion(row):
            emotions = [emotion for emotion in self.emotion_labels if row[emotion] == 1]
            return emotions[0] if emotions else 'neutral'
        
        self.df['primary_emotion'] = self.df.apply(get_primary_emotion, axis=1)
        
        # Create label to index mapping
        unique_emotions = sorted(self.df['primary_emotion'].unique())
        self.label2id = {label: idx for idx, label in enumerate(unique_emotions)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        self.df['label'] = self.df['primary_emotion'].map(self.label2id)
        
        print(f"\n✓ Created single-label dataset")
        print(f"  Number of classes: {len(self.label2id)}")
        print(f"\n  Class distribution:")
        
        class_dist = self.df['primary_emotion'].value_counts()
        for emotion, count in class_dist.head(10).items():
            percentage = (count / len(self.df)) * 100
            print(f"    {emotion:15s}: {count:6,} ({percentage:5.2f}%)")
        
        return self
    
    def split_data(self, test_size=0.15, val_size=0.15, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
        """
        print("\n" + "=" * 80)
        print("SPLITTING DATASET")
        print("=" * 80)
        
        # First split: train+val / test
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['label']
        )
        
        # Second split: train / val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df['label']
        )
        
        print(f"\n✓ Data split complete!")
        print(f"  Training set:   {len(train_df):6,} samples ({len(train_df)/len(self.df)*100:.1f}%)")
        print(f"  Validation set: {len(val_df):6,} samples ({len(val_df)/len(self.df)*100:.1f}%)")
        print(f"  Test set:       {len(test_df):6,} samples ({len(test_df)/len(self.df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df, val_df, test_df, output_dir='processed_data'):
        """Save processed datasets"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n✓ Saving processed data to '{output_dir}/'...")
        
        train_df.to_csv(f'{output_dir}/train.csv', index=False)
        val_df.to_csv(f'{output_dir}/val.csv', index=False)
        test_df.to_csv(f'{output_dir}/test.csv', index=False)
        
        # Save label mappings
        import json
        with open(f'{output_dir}/label_mapping.json', 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label,
                'num_labels': len(self.label2id)
            }, f, indent=2)
        
        print(f"  ✓ train.csv")
        print(f"  ✓ val.csv")
        print(f"  ✓ test.csv")
        print(f"  ✓ label_mapping.json")
        
        return output_dir


def main():
    """Main preprocessing pipeline"""
    print("\n" + "=" * 80)
    print("EMOTION DETECTION - DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Initialize preprocessor
    preprocessor = EmotionDataPreprocessor(
        data_path='go_emotions_dataset (1).csv',
        model_name='distilbert-base-uncased'
    )
    
    # Execute pipeline
    preprocessor.load_data()
    preprocessor.analyze_dataset()
    preprocessor.preprocess_data(remove_unclear=True, remove_no_emotion=True)
    preprocessor.create_single_label_dataset()
    
    # Split and save
    train_df, val_df, test_df = preprocessor.split_data()
    output_dir = preprocessor.save_processed_data(train_df, val_df, test_df)
    
    print("\n" + "=" * 80)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nProcessed data saved to: {output_dir}/")
    print("\nNext step: Run train_emotion_classifier.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
