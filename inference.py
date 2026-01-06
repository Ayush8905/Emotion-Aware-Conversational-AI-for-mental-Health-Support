"""
Inference Script for Emotion Detection
Test the trained model interactively via terminal
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


class EmotionPredictor:
    """
    Emotion detection inference engine
    """
    
    def __init__(self, model_path='models/best_model'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n{'='*80}")
        print(f"EMOTION DETECTION - INFERENCE ENGINE")
        print(f"{'='*80}")
        print(f"  Model: {model_path}")
        print(f"  Device: {self.device}")
        
        # Load model and tokenizer
        print(f"\n  Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        print(f"  ✓ Model loaded successfully!")
        print(f"  Number of emotions: {len(self.id2label)}")
        print(f"{'='*80}\n")
    
    def predict(self, text, top_k=3):
        """
        Predict emotion(s) from text
        
        Args:
            text: Input text
            top_k: Number of top predictions to return
            
        Returns:
            List of (emotion, probability) tuples
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            emotion = self.id2label[idx.item()]
            probability = prob.item()
            results.append((emotion, probability))
        
        return results
    
    def print_prediction(self, text, results):
        """Pretty print prediction results"""
        print(f"\n{'='*80}")
        print(f"INPUT TEXT:")
        print(f"{'='*80}")
        print(f'"{text}"')
        
        print(f"\n{'='*80}")
        print(f"DETECTED EMOTIONS:")
        print(f"{'='*80}")
        
        for i, (emotion, prob) in enumerate(results, 1):
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"{i}. {emotion:15s} {bar} {prob*100:5.2f}%")
        
        print(f"{'='*80}\n")
    
    def interactive_mode(self):
        """Interactive testing mode"""
        print("="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("\nEnter text to detect emotions. Type 'quit' or 'exit' to stop.\n")
        print("Examples:")
        print('  - "I am so happy today!"')
        print('  - "This is really frustrating and annoying."')
        print('  - "I miss you so much."')
        print("\n" + "="*80 + "\n")
        
        while True:
            try:
                text = input("Enter text: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("\n✓ Goodbye!")
                    break
                
                if not text:
                    print("⚠ Please enter some text.\n")
                    continue
                
                results = self.predict(text, top_k=5)
                self.print_prediction(text, results)
                
            except KeyboardInterrupt:
                print("\n\n✓ Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
    
    def batch_test(self, test_texts):
        """Test on a batch of predefined texts"""
        print("="*80)
        print("BATCH TESTING MODE")
        print("="*80)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}/{len(test_texts)}")
            results = self.predict(text, top_k=3)
            self.print_prediction(text, results)


def main():
    """Main inference pipeline"""
    print("\n" + "="*80)
    print("EMOTION DETECTION - INFERENCE")
    print("="*80 + "\n")
    
    # Check if model exists
    model_path = 'models/best_model'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("\nPlease train the model first:")
        print("  1. python data_preprocessing.py")
        print("  2. python train_emotion_classifier.py")
        return
    
    # Initialize predictor
    predictor = EmotionPredictor(model_path)
    
    # Test examples
    print("="*80)
    print("QUICK TEST - SAMPLE PREDICTIONS")
    print("="*80)
    
    test_samples = [
        "I am so excited about this!",
        "This is really disappointing and sad.",
        "I'm feeling quite anxious about the exam.",
        "You did an amazing job, thank you so much!",
        "I'm really angry about what happened.",
        "I love spending time with you.",
        "This is so confusing, I don't understand.",
        "I feel grateful for all your help.",
    ]
    
    predictor.batch_test(test_samples)
    
    # Interactive mode
    print("\n" + "="*80)
    print("Starting interactive mode...")
    print("="*80 + "\n")
    
    predictor.interactive_mode()


if __name__ == "__main__":
    main()
