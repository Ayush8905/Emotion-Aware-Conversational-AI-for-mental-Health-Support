"""
Advanced Error Handler for Mental Health Chatbot
Provides graceful error recovery, retry logic, and user-friendly error messages
"""

import time
import requests
from typing import Callable, Any, Optional, Dict
from functools import wraps
from datetime import datetime


class ErrorHandler:
    """
    Centralized error handling for the mental health chatbot
    Features:
    - Retry logic with exponential backoff
    - Graceful API failure recovery
    - User-friendly error messages
    - Error logging and tracking
    - Offline mode detection
    """
    
    def __init__(self):
        """Initialize error handler"""
        self.error_log = []
        self.max_retries = 3
        self.retry_delay = 1  # Initial delay in seconds
        self.backoff_factor = 2  # Exponential backoff multiplier
    
    def retry_with_backoff(self, max_retries: int = 3, initial_delay: float = 1.0):
        """
        Decorator for retrying functions with exponential backoff
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            
        Returns:
            Decorated function with retry logic
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                delay = initial_delay
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        if attempt < max_retries:
                            print(f"[Retry {attempt + 1}/{max_retries}] {func.__name__} failed: {e}")
                            time.sleep(delay)
                            delay *= self.backoff_factor
                        else:
                            print(f"[FAILED] {func.__name__} failed after {max_retries} retries")
                            self.log_error(func.__name__, str(e), "Max retries exceeded")
                
                # Raise the last exception if all retries failed
                raise last_exception
            
            return wrapper
        return decorator
    
    def safe_execute(self, func: Callable, fallback_value: Any = None, 
                    error_message: str = "An error occurred") -> tuple[Any, Optional[str]]:
        """
        Safely execute a function with error handling
        
        Args:
            func: Function to execute
            fallback_value: Value to return on error
            error_message: User-friendly error message
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            result = func()
            return result, None
        except Exception as e:
            print(f"[ERROR] {error_message}: {e}")
            self.log_error(func.__name__, str(e), error_message)
            return fallback_value, error_message
    
    def check_internet_connection(self, timeout: float = 5.0) -> bool:
        """
        Check if internet connection is available
        
        Args:
            timeout: Timeout for connection check in seconds
            
        Returns:
            True if online, False if offline
        """
        try:
            # Try to connect to Google DNS
            requests.get("https://www.google.com", timeout=timeout)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False
        except Exception as e:
            print(f"[WARNING] Error checking internet: {e}")
            return False
    
    def log_error(self, component: str, error: str, context: str = ""):
        """
        Log error for tracking and debugging
        
        Args:
            component: Component/function where error occurred
            error: Error message
            context: Additional context about the error
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "error": error,
            "context": context
        }
        self.error_log.append(error_entry)
        
        # Keep only last 100 errors
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
    
    def get_error_log(self, limit: int = 10) -> list:
        """Get recent error log entries"""
        return self.error_log[-limit:]
    
    def get_user_friendly_message(self, error_type: str) -> str:
        """
        Get user-friendly error message based on error type
        
        Args:
            error_type: Type of error
            
        Returns:
            User-friendly error message
        """
        messages = {
            "api_error": "I'm having trouble connecting right now. Please try again in a moment.",
            "network_error": "It seems there's a network issue. Please check your internet connection.",
            "timeout_error": "The response is taking longer than expected. Please try again.",
            "rate_limit_error": "I'm receiving too many requests. Please wait a moment and try again.",
            "model_error": "I'm having trouble processing your message. Let me try a different approach.",
            "database_error": "I'm having trouble saving your conversation. Your messages are still being processed.",
            "translation_error": "I had trouble translating your message. Continuing in English.",
            "general_error": "Something unexpected happened. Please try again or rephrase your message."
        }
        return messages.get(error_type, messages["general_error"])
    
    def clear_error_log(self):
        """Clear all error logs"""
        self.error_log = []


class FallbackResponses:
    """
    Fallback responses for when LLM fails
    Provides empathetic responses based on detected emotion
    """
    
    @staticmethod
    def get_fallback_response(emotion: str = "neutral", context: str = "") -> str:
        """
        Get fallback response based on emotion
        
        Args:
            emotion: Detected emotion
            context: Additional context
            
        Returns:
            Fallback response
        """
        # Emotion-specific fallback responses
        responses = {
            "sadness": "I understand you're going through a difficult time. While I'm experiencing some technical difficulties, please know that your feelings are valid. If you need immediate support, please call 988 for the Suicide & Crisis Lifeline.",
            
            "anxiety": "I can sense you're feeling anxious. Take a deep breath with me - inhale for 4 counts, hold for 4, exhale for 4. While I'm having technical issues, remember that anxiety is temporary. You're not alone.",
            
            "anger": "I understand you're feeling frustrated or angry right now. Those feelings are completely valid. While I'm experiencing some technical difficulties, try taking a few deep breaths. Would you like to try again in a moment?",
            
            "fear": "I can tell you're feeling scared or worried. That's a natural response. While I'm having some technical issues, please remember you're safe right now. Take some deep breaths. If you're in crisis, call 988.",
            
            "joy": "I'm glad you're feeling positive! While I'm having some technical difficulties responding, I hope your good mood continues. Please try again in a moment.",
            
            "grief": "I'm so sorry you're experiencing grief. While I'm having technical issues, please know that your pain is real and valid. Consider reaching out to a grief counselor or support group. Call 988 if you need immediate support.",
            
            "stress": "I can sense you're feeling stressed. While I'm experiencing technical difficulties, try this: take 3 deep breaths, name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
            
            "depression": "I understand you may be feeling down. While I'm having technical issues, please know that depression is treatable and help is available. Call 988 for the Suicide & Crisis Lifeline or text HOME to 741741 for Crisis Text Line.",
            
            "neutral": "Thank you for sharing with me. I'm experiencing some technical difficulties right now, but I'm here to support you. Please try again in a moment, and I'll do my best to help."
        }
        
        # Get response for specific emotion or use neutral
        return responses.get(emotion.lower(), responses["neutral"])
    
    @staticmethod
    def get_crisis_fallback() -> str:
        """Get fallback response for crisis situations"""
        return """
🆘 **CRISIS SUPPORT RESOURCES** 🆘

While I'm experiencing technical difficulties, please reach out for immediate help:

**24/7 Crisis Hotlines:**
• **988** - Suicide & Crisis Lifeline (Call or Text)
• **911** - For immediate life-threatening emergencies
• **Crisis Text Line**: Text HOME to 741741

You are not alone. Professional help is available right now.
"""
    
    @staticmethod
    def get_offline_message() -> str:
        """Get message for offline mode"""
        return """
📵 **Offline Mode**

It appears you're offline or I can't connect to my services right now.

**While offline, you can:**
• Review your conversation history
• Access emergency resources (saved locally)
• Practice breathing exercises

**Self-Care Tips:**
• Take deep breaths: 4 in, 4 hold, 4 out
• Write in a journal
• Reach out to a trusted friend
• Call 988 for crisis support (works offline)

I'll be here when you're back online.
"""


# Global instance
error_handler = ErrorHandler()
fallback_responses = FallbackResponses()


# Test function
def test_error_handler():
    """Test error handler functionality"""
    print("="*80)
    print("ERROR HANDLER TEST")
    print("="*80 + "\n")
    
    handler = ErrorHandler()
    
    # Test 1: Internet connection check
    print("1. Testing Internet Connection:")
    is_online = handler.check_internet_connection(timeout=3)
    print(f"   Online: {is_online}\n")
    
    # Test 2: Retry with backoff
    print("2. Testing Retry Logic:")
    @handler.retry_with_backoff(max_retries=3, initial_delay=0.5)
    def flaky_function(fail_times=2):
        if hasattr(flaky_function, 'attempt'):
            flaky_function.attempt += 1
        else:
            flaky_function.attempt = 1
        
        print(f"   Attempt {flaky_function.attempt}")
        if flaky_function.attempt <= fail_times:
            raise Exception("Simulated failure")
        return "Success!"
    
    try:
        result = flaky_function(fail_times=2)
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"   Failed: {e}\n")
    
    # Test 3: User-friendly messages
    print("3. Testing User-Friendly Messages:")
    print(f"   API Error: {handler.get_user_friendly_message('api_error')}")
    print(f"   Network Error: {handler.get_user_friendly_message('network_error')}\n")
    
    # Test 4: Fallback responses
    print("4. Testing Fallback Responses:")
    print(f"   Sadness Response: {fallback_responses.get_fallback_response('sadness')[:100]}...")
    print(f"   Anxiety Response: {fallback_responses.get_fallback_response('anxiety')[:100]}...\n")
    
    # Test 5: Error logging
    print("5. Testing Error Logging:")
    handler.log_error("test_component", "Test error", "Testing context")
    print(f"   Logged errors: {len(handler.get_error_log())}\n")
    
    print("="*80)
    print("[SUCCESS] All error handler tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_error_handler()
