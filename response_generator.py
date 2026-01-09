"""
Empathetic Response Generator using LLAMA 3.2 via Groq API
Generates therapeutic responses based on detected emotions
Phase 4: Response Generation Implementation
"""

import os
from groq import Groq
import json
from typing import Dict, List, Optional
from datetime import datetime


class EmpatheticResponseGenerator:
    """
    Generate empathetic responses using LLAMA 3.2 via Groq API
    Implements Phase 4 of the mental health chatbot project
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize response generator
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY environment variable)
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Please provide Groq API key or set GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"  # Latest LLAMA model
        
        # Emotion-specific prompting strategies
        self.emotion_guidelines = {
            'sadness': 'Show empathy, validate feelings, offer gentle encouragement',
            'anger': 'Acknowledge frustration, help process feelings constructively',
            'fear': 'Provide reassurance, normalize anxiety, suggest coping strategies',
            'nervousness': 'Offer calming techniques, validate concerns, provide structure',
            'joy': 'Share in happiness, encourage positive reflection',
            'gratitude': 'Acknowledge positivity, encourage appreciation practices',
            'grief': 'Show deep empathy, validate loss, provide space for feelings',
            'disappointment': 'Validate feelings, help reframe perspective gently',
            'confusion': 'Help clarify thoughts, break down concerns',
            'embarrassment': 'Normalize human imperfection, offer perspective',
            'love': 'Celebrate positive connections, encourage healthy relationships',
            'excitement': 'Share enthusiasm, encourage positive anticipation',
            'admiration': 'Acknowledge inspiration, encourage positive actions',
            'amusement': 'Share in lighthearted moment, encourage joy',
            'annoyance': 'Validate minor frustrations, help maintain perspective',
            'approval': 'Reinforce positive feelings, encourage confidence',
            'caring': 'Acknowledge compassion, encourage supportive actions',
            'curiosity': 'Encourage exploration, provide thoughtful guidance',
            'desire': 'Validate aspirations, help set healthy goals',
            'disapproval': 'Acknowledge concerns, help process disagreement',
            'disgust': 'Validate strong reactions, help process appropriately',
            'optimism': 'Reinforce positive outlook, encourage hope',
            'pride': 'Celebrate achievements, encourage self-appreciation',
            'realization': 'Acknowledge insight, encourage further reflection',
            'relief': 'Validate release of tension, encourage moving forward',
            'remorse': 'Validate feelings, help with constructive processing',
            'surprise': 'Acknowledge unexpected feelings, help process',
            'neutral': 'Provide supportive, open conversation'
        }
        
        # Crisis detection keywords
        self.crisis_keywords = [
            'kill myself', 'suicide', 'end it all', 'not worth living',
            'want to die', 'self harm', 'hurt myself', 'end my life',
            'better off dead', 'no reason to live', 'can\'t go on'
        ]
    
    def create_system_prompt(self) -> str:
        """Create system prompt for mental health support"""
        return """You are a compassionate mental health support assistant. Your role is to:

1. **Listen actively** and validate the user's emotions
2. **Show empathy** without being patronizing
3. **Provide gentle guidance** and coping strategies when appropriate
4. **Use warm, supportive language**
5. **Avoid:**
   - Giving medical diagnoses
   - Prescribing medications
   - Making promises you can't keep
   - Dismissing feelings with toxic positivity
   - Being overly formal or clinical

6. **Important boundaries:**
   - You are NOT a replacement for professional therapy
   - For crisis situations, recommend professional help immediately
   - Be honest about your limitations
   - Encourage professional help when appropriate

7. **Crisis keywords** (if detected, immediately provide crisis resources):
   - Self-harm, suicide, "end it all", "not worth living"
   
Response style:
- Keep responses 2-4 sentences for emotional validation
- Longer (4-6 sentences) for advice/strategies
- Use "I" statements: "I hear that you're feeling..."
- Ask gentle follow-up questions when appropriate
- Be conversational and natural, not robotic
- Match the user's emotional tone appropriately

Remember: Your goal is to provide emotional support and validation, not therapy or medical advice."""

    def generate_response(
        self, 
        user_message: str, 
        detected_emotion: str, 
        confidence: float,
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        Generate empathetic response based on emotion
        
        Args:
            user_message: User's input text
            detected_emotion: Detected emotion label
            confidence: Emotion detection confidence (0-1)
            conversation_history: Previous conversation turns
            
        Returns:
            Dict with response and metadata
        """
        
        # Check for crisis keywords
        if self._is_crisis(user_message):
            return self._crisis_response()
        
        # Get emotion-specific guidance
        emotion_guidance = self.emotion_guidelines.get(
            detected_emotion.lower(), 
            'Provide supportive, empathetic response'
        )
        
        # Build conversation context
        messages = [{"role": "system", "content": self.create_system_prompt()}]
        
        # Add conversation history if available (last 3 turns for context)
        if conversation_history:
            for turn in conversation_history[-3:]:
                messages.append({"role": "user", "content": turn['user']})
                messages.append({"role": "assistant", "content": turn['assistant']})
        
        # Add current message with emotion context
        emotion_context = f"""User's emotional state: {detected_emotion} (confidence: {confidence:.0%})
Guidance: {emotion_guidance}

User message: {user_message}

Respond with empathy and support. Validate their feelings first, then provide gentle guidance if appropriate."""

        messages.append({"role": "user", "content": emotion_context})
        
        try:
            # Generate response using Groq API
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,  # Balanced creativity
                max_tokens=200,   # Keep responses concise
                top_p=0.9
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            
            return {
                'response': response_text,
                'emotion': detected_emotion,
                'confidence': confidence,
                'model': self.model,
                'success': True,
                'is_crisis': False,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            # Fallback response if API fails
            return {
                'response': f"I'm here to listen and support you. Could you tell me more about what you're experiencing?",
                'emotion': detected_emotion,
                'confidence': confidence,
                'model': self.model,
                'success': False,
                'is_crisis': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _is_crisis(self, text: str) -> bool:
        """Check if message contains crisis keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crisis_keywords)
    
    def _crisis_response(self) -> Dict:
        """
        Emergency response for crisis situations
        """
        crisis_message = """I'm really concerned about what you're sharing. Please know that you don't have to face this alone, and there are people who care and want to help you right now.

🆘 **Immediate Crisis Resources:**

**United States:**
📞 Call/Text: 988 (Suicide & Crisis Lifeline - 24/7)
💬 Chat: 988lifeline.org

**International Crisis Lines:**
• UK: 116 123 (Samaritans)
• Canada: 1-833-456-4566
• Australia: 13 11 14 (Lifeline)
• India: 91-22-27546669 (AASRA)

**Emergency:** Call your local emergency number (911, 112, etc.)

Please reach out to one of these resources right now. Your life has value, and professional counselors are available to help you through this moment. Would you like to talk about reaching out to one of these services?"""

        return {
            'response': crisis_message,
            'emotion': 'CRISIS_DETECTED',
            'confidence': 1.0,
            'model': 'crisis_protocol',
            'success': True,
            'is_crisis': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_with_memory(
        self,
        user_message: str,
        detected_emotion: str,
        confidence: float,
        conversation_memory: List[Dict]
    ) -> Dict:
        """
        Generate response with conversation memory (Phase 5 preview)
        
        Args:
            conversation_memory: List of {user, assistant, emotion, timestamp}
        """
        
        # Track emotional trajectory
        if len(conversation_memory) >= 2:
            recent_emotions = [turn['emotion'] for turn in conversation_memory[-3:]]
            emotion_trend = self._analyze_emotion_trend(recent_emotions)
            
            # Add trend to context
            enhanced_message = f"{user_message}\n[Context: {emotion_trend}]"
        else:
            enhanced_message = user_message
        
        return self.generate_response(
            enhanced_message,
            detected_emotion,
            confidence,
            conversation_memory
        )
    
    def _analyze_emotion_trend(self, emotions: List[str]) -> str:
        """Analyze emotional progression over conversation"""
        negative_emotions = {
            'sadness', 'anger', 'fear', 'nervousness', 'grief', 
            'disappointment', 'annoyance', 'disgust', 'embarrassment'
        }
        positive_emotions = {
            'joy', 'gratitude', 'love', 'excitement', 'relief',
            'admiration', 'amusement', 'optimism', 'pride'
        }
        
        negative_count = sum(1 for e in emotions if e.lower() in negative_emotions)
        positive_count = sum(1 for e in emotions if e.lower() in positive_emotions)
        
        if negative_count > positive_count + 1:
            return "User shows persistent negative emotions - extra support needed"
        elif positive_count > negative_count:
            return "User showing emotional improvement - encourage progress"
        else:
            return "User experiencing mixed emotions - provide balanced support"


def test_response_generator():
    """Test the response generator with sample cases"""
    
    print("\n" + "="*80)
    print("EMPATHETIC RESPONSE GENERATOR - TEST")
    print("="*80 + "\n")
    
    # API key from environment variable
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        api_key = input("Enter your Groq API key: ").strip()
    
    try:
        generator = EmpatheticResponseGenerator(api_key=api_key)
        print("✅ Generator initialized successfully!\n")
        
        # Test cases
        test_cases = [
            {
                'message': "I'm feeling really anxious about my upcoming exam",
                'emotion': 'nervousness',
                'confidence': 0.87
            },
            {
                'message': "I'm so happy! I got the job I wanted!",
                'emotion': 'joy',
                'confidence': 0.92
            },
            {
                'message': "I feel like nobody understands me",
                'emotion': 'sadness',
                'confidence': 0.78
            },
            {
                'message': "Thank you so much for your help!",
                'emotion': 'gratitude',
                'confidence': 0.95
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"TEST CASE {i}")
            print(f"{'='*80}")
            print(f"👤 User: {test['message']}")
            print(f"🎭 Detected Emotion: {test['emotion']} ({test['confidence']:.0%})")
            print(f"\n⏳ Generating response...\n")
            
            result = generator.generate_response(
                test['message'],
                test['emotion'],
                test['confidence']
            )
            
            print(f"🤖 Assistant: {result['response']}")
            print(f"\n✅ Success: {result['success']}")
            
            if not result['success']:
                print(f"⚠️  Error: {result.get('error', 'Unknown error')}")
        
        print(f"\n{'='*80}")
        print("✅ ALL TESTS COMPLETE!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        print("\nPlease check:")
        print("1. Internet connection is active")
        print("2. Groq API key is valid")
        print("3. 'groq' package is installed (pip install groq)")


if __name__ == "__main__":
    test_response_generator()
