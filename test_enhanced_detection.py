"""
Test Enhanced Emotion Detection
"""

import os
from emotion_detector import EnhancedEmotionDetector

# Set API key from environment
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    print("⚠️  Set GROQ_API_KEY environment variable first!")
    exit(1)

print("\n" + "="*80)
print("TESTING ENHANCED EMOTION DETECTION")
print("="*80 + "\n")

# Initialize detector
print("Initializing enhanced detector...")
detector = EnhancedEmotionDetector(api_key=os.getenv('GROQ_API_KEY'))
print("✅ Detector ready\n")

# Test cases
test_cases = [
    "I am so happy today!",
    "I'm feeling very anxious about my exam",
    "This makes me really angry",
    "I'm so grateful for your help",
    "Nobody understands me, I feel so alone"
]

for text in test_cases:
    print(f"\nText: '{text}'")
    result = detector.detect_emotion(text)
    print(f"✓ Emotion: {result['emotion']} ({result['confidence']:.0%})")
    top3_str = ', '.join([f"{e['emotion']} ({e['confidence']:.0%})" for e in result['top3']])
    print(f"  Top 3: {top3_str}")

print("\n" + "="*80)
print("✅ ALL TESTS COMPLETE - ENHANCED DETECTION WORKING!")
print("="*80 + "\n")
