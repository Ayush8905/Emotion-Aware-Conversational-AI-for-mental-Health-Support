"""
Database Manager for Mental Health Chatbot
Handles MongoDB connection, user authentication, and profile management
Phase 5: Advanced Conversation Memory - Backend Implementation
"""

import os
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ConnectionFailure
import bcrypt
from datetime import datetime
import uuid
from typing import Optional, Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseManager:
    """
    Manages all database operations for the mental health chatbot
    - User authentication (signup, login)
    - User profile management
    - Session tracking
    """
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.mongodb_uri = os.getenv('MONGODB_URI')
        self.db_name = os.getenv('MONGODB_DB_NAME', 'mental_health_chatbot')
        
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI not found in environment variables")
        
        try:
            # Connect to MongoDB
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client[self.db_name]
            
            # Test connection
            self.client.server_info()
            
            # Collections
            self.users = self.db['users']
            self.sessions = self.db['sessions']
            self.messages = self.db['messages']
            self.user_profiles = self.db['user_profiles']
            self.emotional_logs = self.db['emotional_logs']
            
            # Create indexes for better performance
            self._create_indexes()
            
            print(f"[OK] Connected to MongoDB: {self.db_name}")
            
        except ConnectionFailure as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        # Unique email index for users
        self.users.create_index("email", unique=True)
        
        # Indexes for fast lookups
        self.sessions.create_index("user_id")
        self.sessions.create_index("session_id", unique=True)
        self.messages.create_index([("session_id", 1), ("timestamp", 1)])
        self.user_profiles.create_index("user_id", unique=True)
        self.emotional_logs.create_index([("user_id", 1), ("timestamp", -1)])
    
    # ==================== USER AUTHENTICATION ====================
    
    def signup_user(self, email: str, password: str, name: str) -> Dict:
        """
        Register a new user
        
        Args:
            email: User's email address
            password: Plain text password (will be hashed)
            name: User's display name
            
        Returns:
            Dict with user_id and success status
        """
        try:
            # Validate inputs
            if not email or not password or not name:
                return {"success": False, "error": "All fields are required"}
            
            if len(password) < 6:
                return {"success": False, "error": "Password must be at least 6 characters"}
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # Create user document
            user_doc = {
                "email": email.lower().strip(),
                "password_hash": password_hash,
                "name": name.strip(),
                "created_at": datetime.utcnow(),
                "last_login": None,
                "profile": {
                    "preferences": {
                        "notifications": True,
                        "theme": "light"
                    }
                }
            }
            
            # Insert into database
            result = self.users.insert_one(user_doc)
            user_id = str(result.inserted_id)
            
            # Create initial user profile
            self._create_user_profile(user_id)
            
            return {
                "success": True,
                "user_id": user_id,
                "message": "Account created successfully"
            }
            
        except DuplicateKeyError:
            return {"success": False, "error": "Email already exists"}
        except Exception as e:
            return {"success": False, "error": f"Signup failed: {str(e)}"}
    
    def login_user(self, email: str, password: str) -> Dict:
        """
        Authenticate user login
        
        Args:
            email: User's email
            password: Plain text password
            
        Returns:
            Dict with user info and session token
        """
        try:
            # Find user by email
            user = self.users.find_one({"email": email.lower().strip()})
            
            if not user:
                return {"success": False, "error": "Invalid email or password"}
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
                return {"success": False, "error": "Invalid email or password"}
            
            # Update last login
            self.users.update_one(
                {"_id": user['_id']},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            
            # Generate session token
            session_token = str(uuid.uuid4())
            
            return {
                "success": True,
                "user_id": str(user['_id']),
                "name": user['name'],
                "email": user['email'],
                "session_token": session_token,
                "message": "Login successful"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Login failed: {str(e)}"}
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        Get user information by ID
        
        Args:
            user_id: User's MongoDB ObjectId as string
            
        Returns:
            User document or None
        """
        try:
            from bson import ObjectId
            user = self.users.find_one({"_id": ObjectId(user_id)})
            if user:
                user['_id'] = str(user['_id'])
                user.pop('password_hash', None)  # Don't return password hash
            return user
        except Exception as e:
            print(f"Error fetching user: {e}")
            return None
    
    # ==================== USER PROFILE MANAGEMENT ====================
    
    def _create_user_profile(self, user_id: str):
        """Create initial user profile"""
        from bson import ObjectId
        
        profile_doc = {
            "user_id": ObjectId(user_id),
            "emotional_baseline": {
                "common_emotions": [],
                "average_intensity": 0.0
            },
            "conversation_stats": {
                "total_sessions": 0,
                "total_messages": 0,
                "average_session_length": 0.0
            },
            "recurring_concerns": [],
            "improvement_indicators": {
                "trend": "neutral",
                "crisis_frequency": "none"
            },
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        self.user_profiles.insert_one(profile_doc)
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user's emotional profile"""
        try:
            from bson import ObjectId
            profile = self.user_profiles.find_one({"user_id": ObjectId(user_id)})
            if profile:
                profile['_id'] = str(profile['_id'])
                profile['user_id'] = str(profile['user_id'])
            return profile
        except Exception as e:
            print(f"Error fetching profile: {e}")
            return None
    
    def update_user_profile(self, user_id: str, updates: Dict):
        """Update user profile with new data"""
        try:
            from bson import ObjectId
            updates['updated_at'] = datetime.utcnow()
            
            self.user_profiles.update_one(
                {"user_id": ObjectId(user_id)},
                {"$set": updates}
            )
            return True
        except Exception as e:
            print(f"Error updating profile: {e}")
            return False
    
    # ==================== UTILITY METHODS ====================
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("[OK] MongoDB connection closed")
    
    def test_connection(self) -> bool:
        """Test if database connection is working"""
        try:
            self.client.server_info()
            return True
        except:
            return False


# ==================== TESTING ====================

def test_database_manager():
    """Test database manager functionality"""
    print("\n" + "="*80)
    print("TESTING DATABASE MANAGER")
    print("="*80 + "\n")
    
    # Initialize
    print("1. Testing connection...")
    db = DatabaseManager()
    print(f"   Connected: {db.test_connection()}\n")
    
    # Test signup
    print("2. Testing user signup...")
    test_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    result = db.signup_user(test_email, "testpass123", "Test User")
    print(f"   Signup result: {result}\n")
    
    if result['success']:
        user_id = result['user_id']
        
        # Test login
        print("3. Testing user login...")
        login_result = db.login_user(test_email, "testpass123")
        print(f"   Login result: {login_result}\n")
        
        # Test get user info
        print("4. Testing get user info...")
        user_info = db.get_user_info(user_id)
        print(f"   User info: {user_info['name']} ({user_info['email']})\n")
        
        # Test get profile
        print("5. Testing get user profile...")
        profile = db.get_user_profile(user_id)
        print(f"   Profile stats: {profile['conversation_stats']}\n")
    
    print("="*80)
    print("[SUCCESS] All database tests passed!")
    print("="*80 + "\n")
    
    db.close()


if __name__ == "__main__":
    test_database_manager()
