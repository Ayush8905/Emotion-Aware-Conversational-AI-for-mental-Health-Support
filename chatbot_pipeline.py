"""
Complete Mental Health Chatbot Pipeline
Combines Emotion Detection (Phase 3) + Response Generation (Phase 4)
Full implementation of empathetic conversational AI
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import time
from datetime import datetime
from response_generator import EmpatheticResponseGenerator
from emotion_detector import EnhancedEmotionDetector
from performance_monitor import performance_monitor


class MentalHealthChatbot:
    """
    Complete mental health support chatbot
    Features:
    - Emotion detection (28 emotions)
    - Empathetic response generation
    - Conversation memory
    - Crisis detection
    """
    
    def __init__(self, emotion_model_path='models/best_model', groq_api_key=None):
        """
        Initialize the complete chatbot pipeline
        
        Args:
            emotion_model_path: Path to trained emotion detection model
            groq_api_key: Groq API key for response generation
        """
        print("\n" + "="*80)
        print("INITIALIZING MENTAL HEALTH CHATBOT")
        print("="*80 + "\n")
        
        # Load enhanced emotion detection system
        print("Loading emotion detection system...")
        try:
            self.emotion_detector = EnhancedEmotionDetector(
                model_path=emotion_model_path,
                api_key=groq_api_key
            )
            print(f"   [OK] Emotion detector ready (28 emotions)")
        except Exception as e:
            print(f"   [ERROR] Error loading emotion detector: {e}")
            raise
        
        # Initialize response generator
        print("Initializing response generator (LLAMA 3.3)...")
        try:
            self.response_generator = EmpatheticResponseGenerator(api_key=groq_api_key)
            print("   [OK] Response generator ready")
        except Exception as e:
            print(f"   [ERROR] Error initializing response generator: {e}")
            raise
        
        # Conversation memory
        self.conversation_history = []
        
        print("\n" + "="*80)
        print("[SUCCESS] CHATBOT INITIALIZATION COMPLETE!")
        print("="*80 + "\n")
    
    def detect_emotion(self, text: str) -> dict:
        """
        Detect emotion in user's text
        
        Args:
            text: User's message
            
        Returns:
            Dict with emotion, confidence, and top-3 predictions
        """
        return self.emotion_detector.detect_emotion(text)
    
    def chat(self, user_message: str, show_emotion: bool = True, username: str = None) -> dict:
        """
        Process user message and generate empathetic response with performance tracking
        
        Args:
            user_message: User's input text
            show_emotion: Whether to display detected emotion
            username: Optional username for performance tracking
            
        Returns:
            Dict with emotion, confidence, response, and metadata
        """
        overall_start_time = time.time()
        
        # Step 1: Detect emotion
        emotion_start_time = time.time()
        emotion_result = self.detect_emotion(user_message)
        emotion_detection_time = time.time() - emotion_start_time
        performance_monitor.log_emotion_detection_time(emotion_detection_time, username)
        
        # Step 2: Generate empathetic response
        llm_start_time = time.time()
        response_result = self.response_generator.generate_response(
            user_message,
            emotion_result['emotion'],
            emotion_result['confidence'],
            self.conversation_history
        )
        llm_response_time = time.time() - llm_start_time
        performance_monitor.log_llm_response_time(llm_response_time, username)
        
        # Step 3: Update conversation history
        conversation_turn = {
            'user': user_message,
            'assistant': response_result['response'],
            'emotion': emotion_result['emotion'],
            'confidence': emotion_result['confidence'],
            'timestamp': datetime.now().isoformat()
        }
        self.conversation_history.append(conversation_turn)
        
        # Log overall response time
        total_response_time = time.time() - overall_start_time
        performance_monitor.log_response_time(total_response_time, username)
        
        return {
            'user_message': user_message,
            'detected_emotion': emotion_result['emotion'],
            'confidence': emotion_result['confidence'],
            'top3_emotions': emotion_result['top3'],
            'response': response_result['response'],
            'is_crisis': response_result.get('is_crisis', False),
            'success': response_result['success'],
            'timestamp': conversation_turn['timestamp'],
            'performance': {
                'emotion_detection_time': round(emotion_detection_time, 3),
                'llm_response_time': round(llm_response_time, 3),
                'total_time': round(total_response_time, 3)
            }
        }
    
    def get_conversation_summary(self) -> dict:
        """Get summary of conversation history"""
        if not self.conversation_history:
            return {'total_turns': 0, 'emotions': []}
        
        emotions = [turn['emotion'] for turn in self.conversation_history]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            'total_turns': len(self.conversation_history),
            'emotions': emotions,
            'emotion_distribution': emotion_counts,
            'start_time': self.conversation_history[0]['timestamp'],
            'last_time': self.conversation_history[-1]['timestamp']
        }
    
    def save_conversation(self, filename: str = None):
        """Save conversation history to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        filepath = os.path.join('conversations', filename)
        os.makedirs('conversations', exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'conversation': self.conversation_history,
                'summary': self.get_conversation_summary()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Conversation saved to: {filepath}")
    
    def interactive_chat(self):
        """
        Interactive chatbot interface with full features
        """
        print("\n" + "="*80)
        print("💬 MENTAL HEALTH SUPPORT CHATBOT - INTERACTIVE MODE")
        print("="*80)
        print("\n👋 Hello! I'm here to listen and support you.")
        print("\n📝 Commands:")
        print("   • Type your message to chat")
        print("   • Type 'quit' or 'exit' to end conversation")
        print("   • Type 'summary' to see conversation summary")
        print("   • Type 'save' to save conversation history")
        print("\n" + "="*80 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n🤖 Assistant: Take care of yourself. Remember, it's okay to seek professional help when you need it. 💙")
                    
                    # Ask if they want to save
                    if self.conversation_history:
                        save_choice = input("\n💾 Would you like to save this conversation? (yes/no): ").strip().lower()
                        if save_choice in ['yes', 'y']:
                            self.save_conversation()
                    
                    print("\nGoodbye! 👋\n")
                    break
                
                elif user_input.lower() == 'summary':
                    summary = self.get_conversation_summary()
                    print("\n" + "="*80)
                    print("📊 CONVERSATION SUMMARY")
                    print("="*80)
                    print(f"Total turns: {summary['total_turns']}")
                    print(f"\nEmotion distribution:")
                    for emotion, count in sorted(summary['emotion_distribution'].items(), 
                                                key=lambda x: x[1], reverse=True):
                        print(f"   • {emotion}: {count}")
                    print("="*80 + "\n")
                    continue
                
                elif user_input.lower() == 'save':
                    self.save_conversation()
                    continue
                
                elif not user_input:
                    continue
                
                # Process message and get response
                print("\n⏳ Processing...\n")
                result = self.chat(user_input)
                
                # Display emotion detection (if not crisis)
                if not result['is_crisis']:
                    print(f"🎭 Detected: {result['detected_emotion']} ({result['confidence']:.0%})")
                    
                    # Show top 3 emotions
                    top3 = result['top3_emotions']
                    if len(top3) > 1:
                        print(f"   Top emotions: ", end="")
                        print(", ".join([f"{e['emotion']} ({e['confidence']:.0%})" 
                                       for e in top3]))
                
                # Display response
                print(f"\n🤖 Assistant: {result['response']}\n")
                print("-" * 80 + "\n")
                
                # Warning if response generation failed
                if not result['success']:
                    print("⚠️  Note: Response generation encountered an issue. Using fallback response.\n")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Conversation interrupted.")
                save_choice = input("💾 Save conversation before exiting? (yes/no): ").strip().lower()
                if save_choice in ['yes', 'y']:
                    self.save_conversation()
                print("\nGoodbye! 👋\n")
                break
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.\n")


def main():
    """Main function to run the chatbot"""
    print("\n" + "="*80)
    print("MENTAL HEALTH CHATBOT - COMPLETE PIPELINE")
    print("Phase 3 (Emotion Detection) + Phase 4 (Response Generation)")
    print("="*80)
    
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("\n[ERROR] Groq API key not found in .env file!")
        print("Please add GROQ_API_KEY to your .env file")
        return
    
    # Initialize chatbot
    try:
        chatbot = MentalHealthChatbot(groq_api_key=api_key)
        
        # Start interactive session
        chatbot.interactive_chat()
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure emotion model exists at 'models/best_model/'")
        print("2. Ensure label mapping exists at 'processed_data/label_mapping.json'")
        print("3. Check internet connection for Groq API")
        print("4. Verify 'groq' package is installed: pip install groq")
        print("\n")


if __name__ == "__main__":
    main()
