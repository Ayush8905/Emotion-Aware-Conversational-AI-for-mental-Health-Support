"""
Conversation Storage Manager
Handles saving, loading, and analyzing conversations in MongoDB
Phase 5: Advanced Conversation Memory - Conversation Persistence
"""

import os
from pymongo import MongoClient
from datetime import datetime
import uuid
from typing import Optional, Dict, List
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()


class ConversationStorage:
    """
    Manages conversation persistence and retrieval
    - Save messages to database
    - Load conversation history
    - Track emotional patterns
    - Session management
    """
    
    def __init__(self):
        """Initialize connection to MongoDB"""
        mongodb_uri = os.getenv('MONGODB_URI')
        db_name = os.getenv('MONGODB_DB_NAME', 'mental_health_chatbot')
        
        if not mongodb_uri:
            raise ValueError("MONGODB_URI not found in environment variables")
        
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        
        # Collections
        self.sessions = self.db['sessions']
        self.messages = self.db['messages']
        self.emotional_logs = self.db['emotional_logs']
        self.user_profiles = self.db['user_profiles']
    
    # ==================== SESSION MANAGEMENT ====================
    
    def create_session(self, user_id: str) -> str:
        """
        Create a new conversation session
        
        Args:
            user_id: User's ID
            
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        
        session_doc = {
            "session_id": session_id,
            "user_id": ObjectId(user_id),
            "start_time": datetime.utcnow(),
            "end_time": None,
            "status": "active",
            "message_count": 0,
            "emotion_summary": {
                "dominant_emotion": None,
                "emotion_distribution": {},
                "average_confidence": 0.0
            }
        }
        
        self.sessions.insert_one(session_doc)
        return session_id
    
    def end_session(self, session_id: str):
        """Mark session as completed and calculate summary"""
        # Get all messages from this session
        messages = list(self.messages.find({"session_id": session_id}))
        
        if not messages:
            return
        
        # Calculate emotion summary
        emotion_counts = {}
        total_confidence = 0
        user_message_count = 0
        
        for msg in messages:
            if msg['role'] == 'user' and msg.get('emotion'):
                emotion = msg['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_confidence += msg.get('confidence', 0)
                user_message_count += 1
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
        average_confidence = total_confidence / user_message_count if user_message_count > 0 else 0
        
        # Update session
        self.sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "end_time": datetime.utcnow(),
                    "status": "completed",
                    "message_count": len(messages),
                    "emotion_summary": {
                        "dominant_emotion": dominant_emotion,
                        "emotion_distribution": emotion_counts,
                        "average_confidence": average_confidence
                    }
                }
            }
        )
        
        # Update user profile stats
        session = self.sessions.find_one({"session_id": session_id})
        if session:
            self._update_user_stats(str(session['user_id']), emotion_counts)
    
    def get_active_session(self, user_id: str) -> Optional[str]:
        """Get user's active session if exists"""
        session = self.sessions.find_one({
            "user_id": ObjectId(user_id),
            "status": "active"
        })
        return session['session_id'] if session else None
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        session = self.sessions.find_one({"session_id": session_id})
        if session:
            session['_id'] = str(session['_id'])
            session['user_id'] = str(session['user_id'])
        return session
    
    # ==================== MESSAGE MANAGEMENT ====================
    
    def save_message(self, session_id: str, user_id: str, role: str, content: str,
                    emotion: str = None, confidence: float = None, 
                    top3_emotions: List = None) -> str:
        """
        Save a message to the database
        
        Args:
            session_id: Session identifier
            user_id: User's ID
            role: 'user' or 'assistant'
            content: Message text
            emotion: Detected emotion (for user messages)
            confidence: Emotion confidence score
            top3_emotions: List of top 3 emotions
            
        Returns:
            message_id: Unique message identifier
        """
        message_doc = {
            "session_id": session_id,
            "user_id": ObjectId(user_id),
            "role": role,
            "content": content,
            "emotion": emotion,
            "confidence": confidence,
            "top3_emotions": top3_emotions or [],
            "timestamp": datetime.utcnow()
        }
        
        result = self.messages.insert_one(message_doc)
        
        # Log emotion if this is a user message
        if role == 'user' and emotion:
            self._log_emotion(user_id, session_id, emotion, confidence)
        
        return str(result.inserted_id)
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message documents
        """
        messages = list(self.messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).limit(limit))
        
        # Convert ObjectIds to strings
        for msg in messages:
            msg['_id'] = str(msg['_id'])
            msg['user_id'] = str(msg['user_id'])
        
        return messages
    
    def get_recent_context(self, session_id: str, num_turns: int = 3) -> List[Dict]:
        """Get last N conversation turns for context"""
        messages = list(self.messages.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(num_turns * 2))
        
        messages.reverse()  # Chronological order
        
        for msg in messages:
            msg['_id'] = str(msg['_id'])
            msg['user_id'] = str(msg['user_id'])
        
        return messages
    
    # ==================== USER HISTORY ====================
    
    def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get all sessions for a user"""
        sessions = list(self.sessions.find(
            {"user_id": ObjectId(user_id)}
        ).sort("start_time", -1).limit(limit))
        
        for session in sessions:
            session['_id'] = str(session['_id'])
            session['user_id'] = str(session['user_id'])
        
        return sessions
    
    def get_user_message_count(self, user_id: str) -> int:
        """Get total message count for user"""
        return self.messages.count_documents({"user_id": ObjectId(user_id)})
    
    # ==================== EMOTIONAL ANALYTICS ====================
    
    def _log_emotion(self, user_id: str, session_id: str, emotion: str, intensity: float):
        """Log emotion for analytics"""
        log_doc = {
            "user_id": ObjectId(user_id),
            "session_id": session_id,
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": datetime.utcnow()
        }
        self.emotional_logs.insert_one(log_doc)
    
    def get_emotional_timeline(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get user's emotional timeline"""
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        logs = list(self.emotional_logs.find({
            "user_id": ObjectId(user_id),
            "timestamp": {"$gte": cutoff_date}
        }).sort("timestamp", 1))
        
        for log in logs:
            log['_id'] = str(log['_id'])
            log['user_id'] = str(log['user_id'])
        
        return logs
    
    def get_emotion_distribution(self, user_id: str) -> Dict:
        """Get user's overall emotion distribution"""
        pipeline = [
            {"$match": {"user_id": ObjectId(user_id)}},
            {"$group": {
                "_id": "$emotion",
                "count": {"$sum": 1},
                "avg_intensity": {"$avg": "$intensity"}
            }},
            {"$sort": {"count": -1}}
        ]
        
        results = list(self.emotional_logs.aggregate(pipeline))
        
        distribution = {}
        for result in results:
            distribution[result['_id']] = {
                "count": result['count'],
                "avg_intensity": round(result['avg_intensity'], 2)
            }
        
        return distribution
    
    def _update_user_stats(self, user_id: str, emotion_counts: Dict):
        """Update user profile with session stats"""
        try:
            # Get current stats
            profile = self.user_profiles.find_one({"user_id": ObjectId(user_id)})
            
            if not profile:
                return
            
            # Update conversation stats
            stats = profile.get('conversation_stats', {})
            stats['total_sessions'] = stats.get('total_sessions', 0) + 1
            
            # Update emotional baseline
            top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            common_emotions = [e[0] for e in top_emotions]
            
            self.user_profiles.update_one(
                {"user_id": ObjectId(user_id)},
                {
                    "$set": {
                        "conversation_stats": stats,
                        "emotional_baseline.common_emotions": common_emotions,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        except Exception as e:
            print(f"Error updating user stats: {e}")
    
    # ==================== EXPORT ====================
    
    def export_conversation(self, session_id: str) -> Dict:
        """Export conversation in JSON format"""
        session = self.get_session_info(session_id)
        messages = self.get_conversation_history(session_id)
        
        return {
            "session_info": session,
            "messages": messages,
            "exported_at": datetime.utcnow().isoformat()
        }
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()


# ==================== TESTING ====================

def test_conversation_storage():
    """Test conversation storage functionality"""
    print("\n" + "="*80)
    print("TESTING CONVERSATION STORAGE")
    print("="*80 + "\n")
    
    # Note: This requires a valid user_id from database_manager
    # For testing, we'll use a dummy ObjectId
    
    storage = ConversationStorage()
    
    print("1. Testing session creation...")
    # You'll need a real user_id from your database
    test_user_id = "507f1f77bcf86cd799439011"  # Example ObjectId
    
    try:
        session_id = storage.create_session(test_user_id)
        print(f"   Created session: {session_id}\n")
        
        print("2. Testing message saving...")
        storage.save_message(
            session_id, test_user_id, "user",
            "I'm feeling anxious", "anxiety", 0.87
        )
        storage.save_message(
            session_id, test_user_id, "assistant",
            "I understand you're feeling anxious..."
        )
        print("   Messages saved successfully\n")
        
        print("3. Testing conversation history...")
        history = storage.get_conversation_history(session_id)
        print(f"   Retrieved {len(history)} messages\n")
        
        print("4. Testing session end...")
        storage.end_session(session_id)
        session_info = storage.get_session_info(session_id)
        print(f"   Session status: {session_info['status']}\n")
        
        print("="*80)
        print("[SUCCESS] All conversation storage tests passed!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"   [ERROR] Test failed: {e}\n")
    
    storage.close()


if __name__ == "__main__":
    test_conversation_storage()
