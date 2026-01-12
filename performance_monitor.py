"""
Performance Monitor for Phase 7 User Study
Tracks system performance metrics and response times
"""

import time
from datetime import datetime
from typing import Dict, Optional
from feedback_system import feedback_system
import psutil
import platform

class PerformanceMonitor:
    """Monitors and logs system performance metrics"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.start_time = time.time()
        self.request_count = 0
        self.total_response_time = 0
        self.error_count = 0
    
    def log_response_time(self, response_time: float, username: Optional[str] = None):
        """
        Log response time for a chatbot interaction
        
        Args:
            response_time: Time taken to generate response in seconds
            username: Optional username
        """
        self.request_count += 1
        self.total_response_time += response_time
        
        # Log to database
        feedback_system.log_performance_metric(
            metric_type='response_time',
            value=response_time,
            username=username,
            details={
                'timestamp': datetime.utcnow().isoformat(),
                'request_count': self.request_count
            }
        )
    
    def log_emotion_detection_time(self, detection_time: float, username: Optional[str] = None):
        """
        Log emotion detection processing time
        
        Args:
            detection_time: Time taken for emotion detection in seconds
            username: Optional username
        """
        feedback_system.log_performance_metric(
            metric_type='emotion_detection_time',
            value=detection_time,
            username=username,
            details={'timestamp': datetime.utcnow().isoformat()}
        )
    
    def log_llm_response_time(self, llm_time: float, username: Optional[str] = None):
        """
        Log LLM response generation time
        
        Args:
            llm_time: Time taken for LLM response in seconds
            username: Optional username
        """
        feedback_system.log_performance_metric(
            metric_type='llm_response_time',
            value=llm_time,
            username=username,
            details={'timestamp': datetime.utcnow().isoformat()}
        )
    
    def log_error(self, error_type: str, error_message: str, username: Optional[str] = None):
        """
        Log system errors
        
        Args:
            error_type: Type of error (e.g., 'emotion_detection', 'llm_generation')
            error_message: Error message
            username: Optional username
        """
        self.error_count += 1
        
        feedback_system.log_performance_metric(
            metric_type='error',
            value=1,
            username=username,
            details={
                'error_type': error_type,
                'error_message': error_message,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def get_system_stats(self) -> Dict:
        """
        Get current system statistics
        
        Returns:
            Dictionary with system stats
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage': cpu_percent,
                'memory_total': memory.total / (1024**3),  # GB
                'memory_used': memory.used / (1024**3),  # GB
                'memory_percent': memory.percent,
                'disk_total': disk.total / (1024**3),  # GB
                'disk_used': disk.used / (1024**3),  # GB
                'disk_percent': disk.percent,
                'platform': platform.system(),
                'python_version': platform.python_version()
            }
        except Exception as e:
            print(f"[ERROR] Failed to get system stats: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict:
        """
        Get summary of performance metrics
        
        Returns:
            Dictionary with performance summary
        """
        uptime = time.time() - self.start_time
        avg_response_time = (self.total_response_time / self.request_count) if self.request_count > 0 else 0
        
        return {
            'uptime_seconds': round(uptime, 2),
            'uptime_minutes': round(uptime / 60, 2),
            'uptime_hours': round(uptime / 3600, 2),
            'total_requests': self.request_count,
            'average_response_time': round(avg_response_time, 3),
            'total_errors': self.error_count,
            'error_rate': round((self.error_count / self.request_count * 100) if self.request_count > 0 else 0, 2)
        }
    
    def log_user_session_start(self, username: str):
        """
        Log when a user starts a session
        
        Args:
            username: Username
        """
        feedback_system.log_performance_metric(
            metric_type='session_start',
            value=1,
            username=username,
            details={'timestamp': datetime.utcnow().isoformat()}
        )
    
    def log_user_session_end(self, username: str, duration: float, message_count: int):
        """
        Log when a user ends a session
        
        Args:
            username: Username
            duration: Session duration in seconds
            message_count: Number of messages exchanged
        """
        feedback_system.log_performance_metric(
            metric_type='session_end',
            value=duration,
            username=username,
            details={
                'duration_seconds': duration,
                'message_count': message_count,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.start_time = time.time()
        self.request_count = 0
        self.total_response_time = 0
        self.error_count = 0


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Test function
if __name__ == "__main__":
    print("=== Performance Monitor Test ===\n")
    
    # Simulate some performance logging
    performance_monitor.log_response_time(1.25, username="test_user")
    performance_monitor.log_response_time(0.95, username="test_user")
    performance_monitor.log_response_time(1.10, username="test_user")
    
    performance_monitor.log_emotion_detection_time(0.15)
    performance_monitor.log_llm_response_time(0.85)
    
    performance_monitor.log_error("llm_generation", "API timeout", username="test_user")
    
    # Get summary
    summary = performance_monitor.get_performance_summary()
    print("Performance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get system stats
    print("\nSystem Stats:")
    stats = performance_monitor.get_system_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
