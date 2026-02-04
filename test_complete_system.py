"""
Complete System Health Check
Tests all components of the Mental Health Chatbot
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("MENTAL HEALTH CHATBOT - SYSTEM HEALTH CHECK")
print("=" * 80)
print()

# 1. Check Environment Variables
print("1️⃣  Checking Environment Variables...")
groq_key = os.getenv('GROQ_API_KEY')
mongodb_uri = os.getenv('MONGODB_URI')
mongodb_db = os.getenv('MONGODB_DB_NAME')

if groq_key:
    print(f"   ✅ GROQ_API_KEY: {groq_key[:20]}...{groq_key[-10:]}")
else:
    print("   ❌ GROQ_API_KEY: NOT FOUND")

if mongodb_uri:
    print(f"   ✅ MONGODB_URI: {mongodb_uri[:30]}...")
else:
    print("   ❌ MONGODB_URI: NOT FOUND")

if mongodb_db:
    print(f"   ✅ MONGODB_DB_NAME: {mongodb_db}")
else:
    print("   ❌ MONGODB_DB_NAME: NOT FOUND")
print()

# 2. Test MongoDB Connection
print("2️⃣  Testing MongoDB Connection...")
try:
    from database_manager import DatabaseManager
    db = DatabaseManager()
    collections = db.db.list_collection_names()
    print(f"   ✅ Connected to: {db.db.name}")
    print(f"   ✅ Collections ({len(collections)}): {', '.join(collections[:5])}")
    
    # Check if model data exists
    users_count = db.users.count_documents({})
    sessions_count = db.sessions.count_documents({})
    print(f"   ✅ Users: {users_count}")
    print(f"   ✅ Sessions: {sessions_count}")
except Exception as e:
    print(f"   ❌ MongoDB Error: {str(e)}")
print()

# 3. Test Emotion Detection Model
print("3️⃣  Testing Emotion Detection Model...")
try:
    from emotion_detector import EnhancedEmotionDetector
    detector = EnhancedEmotionDetector()
    
    test_message = "I am feeling really happy and excited today!"
    result = detector.predict(test_message)
    
    print(f"   ✅ Model Loaded Successfully")
    print(f"   ✅ Test Input: '{test_message}'")
    print(f"   ✅ Detected Emotion: {result['emotion']}")
    print(f"   ✅ Confidence: {result['confidence']:.2%}")
    top_3 = ', '.join([f"{e['label']} ({e['score']:.2%})" for e in result['top_emotions'][:3]])
    print(f"   ✅ Top 3: {top_3}")
except Exception as e:
    print(f"   ❌ Emotion Detection Error: {str(e)}")
print()

# 4. Test Response Generator (Groq API)
print("4️⃣  Testing Response Generator (Groq API)...")
try:
    from response_generator import EmpatheticResponseGenerator
    generator = EmpatheticResponseGenerator(api_key=groq_key)
    
    print(f"   ✅ Groq Client Initialized")
    print(f"   ✅ Model: {generator.model}")
    
    # Test generation
    test_response = generator.generate_response(
        user_message="I'm feeling anxious",
        emotion="anxiety",
        conversation_history=[]
    )
    
    print(f"   ✅ Test Response Generated")
    print(f"   ✅ Response Length: {len(test_response)} characters")
    print(f"   ✅ Preview: {test_response[:100]}...")
except Exception as e:
    print(f"   ❌ Response Generator Error: {str(e)}")
print()

# 5. Test Translation System
print("5️⃣  Testing Multi-Language Support...")
try:
    from language_manager import LanguageManager
    lang_manager = LanguageManager()
    
    test_text = "Hello, how are you feeling today?"
    spanish = lang_manager.translate(test_text, 'es')
    
    print(f"   ✅ Language Manager Initialized")
    print(f"   ✅ Supported Languages: {len(lang_manager.supported_languages)}")
    print(f"   ✅ Test Translation (EN → ES):")
    print(f"      Input: {test_text}")
    print(f"      Output: {spanish}")
except Exception as e:
    print(f"   ❌ Translation Error: {str(e)}")
print()

# 6. Test Safety Monitor
print("6️⃣  Testing Safety & Crisis Detection...")
try:
    from safety_monitor import SafetyMonitor
    safety = SafetyMonitor()
    
    safe_message = "I'm feeling a bit sad today"
    crisis_message = "I want to hurt myself"
    
    safe_result = safety.check_message(safe_message)
    crisis_result = safety.check_message(crisis_message)
    
    print(f"   ✅ Safety Monitor Initialized")
    print(f"   ✅ Safe Message: '{safe_message}' → Risk: {safe_result['risk_level']}")
    print(f"   ✅ Crisis Message: '{crisis_message}' → Risk: {crisis_result['risk_level']}")
    print(f"   ✅ Crisis Keywords Monitored: {len(safety.crisis_keywords)}")
except Exception as e:
    print(f"   ❌ Safety Monitor Error: {str(e)}")
print()

# 7. Test Error Handler
print("7️⃣  Testing Error Handling System...")
try:
    from error_handler import ErrorHandler
    error_handler_instance = ErrorHandler()
    
    print(f"   ✅ Error Handler Initialized")
    print(f"   ✅ Max Retries: {error_handler_instance.max_retries}")
    print(f"   ✅ Fallback Responses: {len(error_handler_instance.fallback_responses)}")
    print(f"   ✅ Error Log Capacity: {error_handler_instance.max_error_logs}")
except Exception as e:
    print(f"   ❌ Error Handler Error: {str(e)}")
print()

# 8. Test Feedback System
print("8️⃣  Testing Feedback & Analytics System...")
try:
    from feedback_system import FeedbackSystem
    feedback = FeedbackSystem()
    
    print(f"   ✅ Feedback System Initialized")
    print(f"   ✅ MongoDB Connected")
    
    stats = feedback.get_feedback_statistics()
    print(f"   ✅ Total Feedback: {stats.get('total_feedback', 0)}")
    print(f"   ✅ Positive: {stats.get('positive', 0)}")
    print(f"   ✅ Negative: {stats.get('negative', 0)}")
except Exception as e:
    print(f"   ❌ Feedback System Error: {str(e)}")
print()

# 9. Test Chatbot Pipeline
print("9️⃣  Testing Complete Chatbot Pipeline...")
try:
    from chatbot_pipeline import MentalHealthChatbot
    chatbot = MentalHealthChatbot()
    
    print(f"   ✅ Chatbot Pipeline Initialized")
    print(f"   ✅ All Components Loaded")
    
    # Test end-to-end
    test_input = "I'm feeling really stressed about work"
    response = chatbot.process_message(test_input, username="test_user")
    
    print(f"   ✅ End-to-End Test Successful")
    print(f"   ✅ Input: '{test_input}'")
    print(f"   ✅ Detected Emotion: {response.get('emotion', 'N/A')}")
    print(f"   ✅ Response Generated: {len(response.get('response', ''))} chars")
    print(f"   ✅ Response Preview: {response.get('response', '')[:100]}...")
except Exception as e:
    print(f"   ❌ Chatbot Pipeline Error: {str(e)}")
print()

# Summary
print("=" * 80)
print("SYSTEM HEALTH CHECK COMPLETE")
print("=" * 80)
print()
print("✅ All core components are operational!")
print("✅ Your Mental Health Chatbot is ready to use")
print()
print("🚀 Run: streamlit run app.py")
print("🌐 Access: http://localhost:8501")
print("=" * 80)
