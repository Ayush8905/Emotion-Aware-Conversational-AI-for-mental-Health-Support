"""
Quick test script for chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from chatbot_pipeline import MentalHealthChatbot

print("\n" + "="*80)
print("TESTING MENTAL HEALTH CHATBOT")
print("="*80 + "\n")

# Initialize chatbot
bot = MentalHealthChatbot()

# Test cases
test_messages = [
    "I'm feeling very anxious about my exam tomorrow",
    "I feel so happy today!",
    "Nobody understands me, I feel alone"
]

for msg in test_messages:
    print("\n" + "="*80)
    print(f"YOU: {msg}")
    print("="*80)
    
    result = bot.chat(msg, show_emotion=True)
    
    print(f"\nEMOTION: {result['detected_emotion']} ({result['confidence']:.0%})")
    print(f"CHATBOT: {result['response']}")
    print("="*80)
