"""
Feedback System for User Study & Validation (Phase 7)
Collects user feedback on chatbot responses and system performance
"""

from datetime import datetime
from pymongo import MongoClient, DESCENDING
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

class FeedbackSystem:
    """Manages user feedback collection and storage"""
    
    def __init__(self):
        """Initialize feedback system with MongoDB connection"""
        self.mongodb_uri = os.getenv('MONGODB_URI', 'mongodb+srv://Ayush:k8f6Sh3OLmPaJ4Ay@cluster0.opgn0zz.mongodb.net/')
        self.database_name = os.getenv('MONGODB_DATABASE', 'mental_health_chatbot')
        
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.database_name]
            self.feedback_collection = self.db['feedback']
            self.surveys_collection = self.db['surveys']
            self.performance_logs = self.db['performance_logs']
            
            # Test connection
            self.client.admin.command('ping')
            print("[OK] Feedback System connected to MongoDB")
            
            # Create indexes for efficient queries
            self._create_indexes()
            
        except Exception as e:
            print(f"[ERROR] Failed to connect feedback system to MongoDB: {e}")
            self.client = None
            self.db = None
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        try:
            # Index for feedback queries
            self.feedback_collection.create_index([("username", 1), ("timestamp", DESCENDING)])
            self.feedback_collection.create_index([("conversation_id", 1)])
            self.feedback_collection.create_index([("rating", 1)])
            
            # Index for survey queries
            self.surveys_collection.create_index([("username", 1), ("timestamp", DESCENDING)])
            
            # Index for performance logs
            self.performance_logs.create_index([("timestamp", DESCENDING)])
            
        except Exception as e:
            print(f"[WARNING] Could not create indexes: {e}")
    
    def record_response_feedback(
        self,
        username: str,
        conversation_id: str,
        message_index: int,
        user_message: str,
        bot_response: str,
        detected_emotion: str,
        rating: str,  # 'positive', 'negative', 'neutral'
        comment: Optional[str] = None
    ) -> bool:
        """
        Record user feedback on a specific chatbot response
        
        Args:
            username: User who provided feedback
            conversation_id: ID of the conversation
            message_index: Index of the message in conversation
            user_message: Original user message
            bot_response: Chatbot's response
            detected_emotion: Emotion detected in user message
            rating: positive/negative/neutral
            comment: Optional text comment
        
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            print("[ERROR] No database connection")
            return False
        
        try:
            feedback_data = {
                'username': username,
                'conversation_id': conversation_id,
                'message_index': message_index,
                'user_message': user_message,
                'bot_response': bot_response,
                'detected_emotion': detected_emotion,
                'rating': rating,
                'comment': comment,
                'timestamp': datetime.utcnow(),
                'feedback_type': 'response_rating'
            }
            
            # Check if feedback already exists for this message
            existing = self.feedback_collection.find_one({
                'username': username,
                'conversation_id': conversation_id,
                'message_index': message_index
            })
            
            if existing:
                # Update existing feedback
                self.feedback_collection.update_one(
                    {'_id': existing['_id']},
                    {'$set': feedback_data}
                )
            else:
                # Insert new feedback
                self.feedback_collection.insert_one(feedback_data)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to record feedback: {e}")
            return False
    
    def record_satisfaction_survey(
        self,
        username: str,
        conversation_id: str,
        overall_satisfaction: int,  # 1-5 scale
        empathy_rating: int,  # 1-5 scale
        helpfulness_rating: int,  # 1-5 scale
        ease_of_use: int,  # 1-5 scale
        would_recommend: bool,
        comments: Optional[str] = None,
        suggestions: Optional[str] = None
    ) -> bool:
        """
        Record user satisfaction survey after conversation
        
        Args:
            username: User completing survey
            conversation_id: ID of the conversation
            overall_satisfaction: 1-5 rating
            empathy_rating: 1-5 rating
            helpfulness_rating: 1-5 rating
            ease_of_use: 1-5 rating
            would_recommend: True/False
            comments: Optional comments
            suggestions: Optional suggestions for improvement
        
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            print("[ERROR] No database connection")
            return False
        
        try:
            survey_data = {
                'username': username,
                'conversation_id': conversation_id,
                'ratings': {
                    'overall_satisfaction': overall_satisfaction,
                    'empathy': empathy_rating,
                    'helpfulness': helpfulness_rating,
                    'ease_of_use': ease_of_use
                },
                'would_recommend': would_recommend,
                'comments': comments,
                'suggestions': suggestions,
                'timestamp': datetime.utcnow(),
                'survey_type': 'post_conversation'
            }
            
            self.surveys_collection.insert_one(survey_data)
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to record survey: {e}")
            return False
    
    def log_performance_metric(
        self,
        metric_type: str,
        value: float,
        username: Optional[str] = None,
        details: Optional[Dict] = None
    ) -> bool:
        """
        Log system performance metrics
        
        Args:
            metric_type: Type of metric (response_time, emotion_accuracy, etc.)
            value: Metric value
            username: Optional username
            details: Optional additional details
        
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            log_data = {
                'metric_type': metric_type,
                'value': value,
                'username': username,
                'details': details or {},
                'timestamp': datetime.utcnow()
            }
            
            self.performance_logs.insert_one(log_data)
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to log performance: {e}")
            return False
    
    def get_feedback_statistics(self) -> Dict:
        """
        Get overall feedback statistics
        
        Returns:
            Dictionary with feedback statistics
        """
        if not self.client:
            return {}
        
        try:
            # Count feedback by rating
            total_feedback = self.feedback_collection.count_documents({})
            positive = self.feedback_collection.count_documents({'rating': 'positive'})
            negative = self.feedback_collection.count_documents({'rating': 'negative'})
            neutral = self.feedback_collection.count_documents({'rating': 'neutral'})
            
            # Calculate satisfaction rate
            satisfaction_rate = (positive / total_feedback * 100) if total_feedback > 0 else 0
            
            # Get survey statistics
            surveys = list(self.surveys_collection.find())
            avg_satisfaction = 0
            avg_empathy = 0
            avg_helpfulness = 0
            avg_ease_of_use = 0
            recommend_count = 0
            
            if surveys:
                avg_satisfaction = sum(s['ratings']['overall_satisfaction'] for s in surveys) / len(surveys)
                avg_empathy = sum(s['ratings']['empathy'] for s in surveys) / len(surveys)
                avg_helpfulness = sum(s['ratings']['helpfulness'] for s in surveys) / len(surveys)
                avg_ease_of_use = sum(s['ratings']['ease_of_use'] for s in surveys) / len(surveys)
                recommend_count = sum(1 for s in surveys if s['would_recommend'])
            
            return {
                'total_feedback': total_feedback,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'satisfaction_rate': round(satisfaction_rate, 2),
                'total_surveys': len(surveys),
                'avg_satisfaction': round(avg_satisfaction, 2),
                'avg_empathy': round(avg_empathy, 2),
                'avg_helpfulness': round(avg_helpfulness, 2),
                'avg_ease_of_use': round(avg_ease_of_use, 2),
                'recommend_rate': round((recommend_count / len(surveys) * 100) if surveys else 0, 2)
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to get statistics: {e}")
            return {}
    
    def get_user_feedback_history(self, username: str, limit: int = 50) -> List[Dict]:
        """
        Get feedback history for a specific user
        
        Args:
            username: Username to query
            limit: Maximum number of records
        
        Returns:
            List of feedback records
        """
        if not self.client:
            return []
        
        try:
            feedback = list(self.feedback_collection.find(
                {'username': username}
            ).sort('timestamp', DESCENDING).limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for item in feedback:
                item['_id'] = str(item['_id'])
            
            return feedback
            
        except Exception as e:
            print(f"[ERROR] Failed to get user feedback: {e}")
            return []
    
    def get_recent_feedback(self, limit: int = 20) -> List[Dict]:
        """
        Get most recent feedback across all users
        
        Args:
            limit: Maximum number of records
        
        Returns:
            List of recent feedback
        """
        if not self.client:
            return []
        
        try:
            feedback = list(self.feedback_collection.find()
                          .sort('timestamp', DESCENDING)
                          .limit(limit))
            
            for item in feedback:
                item['_id'] = str(item['_id'])
            
            return feedback
            
        except Exception as e:
            print(f"[ERROR] Failed to get recent feedback: {e}")
            return []
    
    def export_feedback_data(self, output_format: str = 'dict') -> List[Dict]:
        """
        Export all feedback data for analysis
        
        Args:
            output_format: Format for export (dict, csv)
        
        Returns:
            List of all feedback records
        """
        if not self.client:
            return []
        
        try:
            all_feedback = list(self.feedback_collection.find())
            
            for item in all_feedback:
                item['_id'] = str(item['_id'])
                # Format timestamp as string
                if 'timestamp' in item:
                    item['timestamp'] = item['timestamp'].isoformat()
            
            return all_feedback
            
        except Exception as e:
            print(f"[ERROR] Failed to export feedback: {e}")
            return []
    
    def get_emotion_feedback_breakdown(self) -> Dict:
        """
        Get feedback breakdown by detected emotion
        
        Returns:
            Dictionary mapping emotions to feedback counts
        """
        if not self.client:
            return {}
        
        try:
            pipeline = [
                {
                    '$group': {
                        '_id': '$detected_emotion',
                        'positive': {
                            '$sum': {
                                '$cond': [{'$eq': ['$rating', 'positive']}, 1, 0]
                            }
                        },
                        'negative': {
                            '$sum': {
                                '$cond': [{'$eq': ['$rating', 'negative']}, 1, 0]
                            }
                        },
                        'neutral': {
                            '$sum': {
                                '$cond': [{'$eq': ['$rating', 'neutral']}, 1, 0]
                            }
                        },
                        'total': {'$sum': 1}
                    }
                },
                {
                    '$sort': {'total': -1}
                }
            ]
            
            results = list(self.feedback_collection.aggregate(pipeline))
            
            emotion_breakdown = {}
            for result in results:
                emotion = result['_id'] or 'unknown'
                emotion_breakdown[emotion] = {
                    'positive': result['positive'],
                    'negative': result['negative'],
                    'neutral': result['neutral'],
                    'total': result['total'],
                    'satisfaction_rate': round((result['positive'] / result['total'] * 100), 2)
                }
            
            return emotion_breakdown
            
        except Exception as e:
            print(f"[ERROR] Failed to get emotion breakdown: {e}")
            return {}
    
    def get_average_response_time(self) -> float:
        """
        Get average response time from performance logs
        
        Returns:
            Average response time in seconds
        """
        if not self.client:
            return 0.0
        
        try:
            pipeline = [
                {
                    '$match': {'metric_type': 'response_time'}
                },
                {
                    '$group': {
                        '_id': None,
                        'avg_time': {'$avg': '$value'}
                    }
                }
            ]
            
            result = list(self.performance_logs.aggregate(pipeline))
            
            if result:
                return round(result[0]['avg_time'], 3)
            return 0.0
            
        except Exception as e:
            print(f"[ERROR] Failed to get average response time: {e}")
            return 0.0
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()


# Global feedback system instance
feedback_system = FeedbackSystem()


# Test function
if __name__ == "__main__":
    print("=== Feedback System Test ===\n")
    
    # Test feedback recording
    success = feedback_system.record_response_feedback(
        username="test_user",
        conversation_id="test_conv_123",
        message_index=0,
        user_message="I'm feeling anxious",
        bot_response="I hear that you're feeling anxious...",
        detected_emotion="nervousness",
        rating="positive",
        comment="Very helpful response"
    )
    print(f"Feedback recorded: {success}")
    
    # Test survey recording
    success = feedback_system.record_satisfaction_survey(
        username="test_user",
        conversation_id="test_conv_123",
        overall_satisfaction=5,
        empathy_rating=5,
        helpfulness_rating=4,
        ease_of_use=5,
        would_recommend=True,
        comments="Great chatbot!",
        suggestions="Could add more coping strategies"
    )
    print(f"Survey recorded: {success}")
    
    # Test statistics
    stats = feedback_system.get_feedback_statistics()
    print(f"\nFeedback Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test emotion breakdown
    print(f"\nEmotion Feedback Breakdown:")
    breakdown = feedback_system.get_emotion_feedback_breakdown()
    for emotion, data in breakdown.items():
        print(f"  {emotion}: {data}")
    
    feedback_system.close()
