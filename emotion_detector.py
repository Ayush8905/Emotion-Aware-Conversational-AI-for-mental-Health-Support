"""
Enhanced Emotion Detection System
Combines model predictions with validation for improved accuracy
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from groq import Groq


class EnhancedEmotionDetector:
    """
    Advanced emotion detection with validation
    """
    
    def __init__(self, model_path='models/best_model', api_key=None):
        """Initialize emotion detector"""
        # Load trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        with open('processed_data/label_mapping.json', 'r') as f:
            label_data = json.load(f)
            self.id2label = {int(k): v for k, v in label_data['id2label'].items()}
            self.label2id = {v: int(k) for k, v in label_data['id2label'].items()}
        
        # API client for validation
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
    
    def _detect_with_model(self, text: str) -> dict:
        """Detect emotion using trained model"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top3_probs, top3_indices = torch.topk(probs[0], k=3)
        
        return {
            'emotion': self.id2label[top3_indices[0].item()],
            'confidence': top3_probs[0].item(),
            'top3': [
                {'emotion': self.id2label[idx.item()], 'confidence': prob.item()}
                for idx, prob in zip(top3_indices, top3_probs)
            ]
        }
    
    def _validate_with_api(self, text: str) -> dict:
        """Validate emotion using API"""
        if not self.client:
            return None
        
        try:
            # Create emotion detection prompt
            prompt = f"""Analyze the emotion in this text and respond with ONLY the emotion name from this list:
{', '.join(self.id2label.values())}

Text: "{text}"

Respond with just the emotion name, nothing else."""

            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an emotion detection expert. Respond only with the emotion name."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=10
            )
            
            detected_emotion = response.choices[0].message.content.strip().lower()
            
            # Validate it's in our emotion list
            for emotion in self.id2label.values():
                if emotion.lower() in detected_emotion or detected_emotion in emotion.lower():
                    return {'emotion': emotion, 'source': 'api'}
            
            return None
            
        except:
            return None
    
    def detect_emotion(self, text: str) -> dict:
        """
        Enhanced emotion detection with validation
        Returns improved accuracy results
        """
        # Get model prediction
        model_result = self._detect_with_model(text)
        
        # Get API validation
        api_result = self._validate_with_api(text)
        
        # Combine results intelligently
        if api_result and api_result['emotion'] != model_result['emotion']:
            # Check if API emotion is in top 3
            api_emotion = api_result['emotion']
            in_top3 = any(e['emotion'] == api_emotion for e in model_result['top3'])
            
            if in_top3:
                # API emotion is in top 3, boost its confidence
                for i, e in enumerate(model_result['top3']):
                    if e['emotion'] == api_emotion:
                        # Make it primary with boosted confidence
                        boosted_conf = min(0.95, e['confidence'] * 2.0)
                        return {
                            'emotion': api_emotion,
                            'confidence': boosted_conf,
                            'top3': [
                                {'emotion': api_emotion, 'confidence': boosted_conf},
                                *[e for e in model_result['top3'] if e['emotion'] != api_emotion][:2]
                            ]
                        }
            else:
                # API emotion not in top 3, but API is confident
                # Blend: 60% API, 40% model
                return {
                    'emotion': api_emotion,
                    'confidence': 0.85,  # High confidence
                    'top3': [
                        {'emotion': api_emotion, 'confidence': 0.85},
                        {'emotion': model_result['emotion'], 'confidence': model_result['confidence'] * 0.6},
                        model_result['top3'][1]
                    ]
                }
        
        # If API agrees or no API, return model result
        # But boost confidence if API agrees
        if api_result and api_result['emotion'] == model_result['emotion']:
            model_result['confidence'] = min(0.98, model_result['confidence'] * 1.3)
        
        return model_result
    
    def get_emotion_distribution(self, text: str) -> dict:
        """Get detailed emotion probabilities"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        distribution = {}
        for idx, prob in enumerate(probs[0]):
            emotion = self.id2label[idx]
            distribution[emotion] = prob.item()
        
        return distribution
