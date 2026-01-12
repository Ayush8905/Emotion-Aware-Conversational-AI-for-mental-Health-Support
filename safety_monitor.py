"""
Safety Monitor - Crisis Detection & Content Safety
Detects harmful content and provides emergency resources
"""

import re
from typing import Dict, List, Tuple

class SafetyMonitor:
    """Monitors conversations for safety concerns and crisis indicators"""
    
    def __init__(self):
        # Crisis keywords and patterns
        self.crisis_keywords = {
            'suicide': ['suicide', 'kill myself', 'end my life', 'want to die', 'better off dead', 
                       'no reason to live', 'suicidal', 'take my own life'],
            'self_harm': ['cut myself', 'hurt myself', 'self harm', 'self-harm', 'cutting', 
                         'burning myself', 'harm myself'],
            'violence': ['hurt someone', 'kill someone', 'harm others', 'violent thoughts', 
                        'want to hurt'],
            'abuse': ['being abused', 'someone hurts me', 'physical abuse', 'sexual abuse', 
                     'domestic violence', 'being harmed']
        }
        
        # Warning keywords (less severe)
        self.warning_keywords = [
            'depressed', 'hopeless', 'worthless', 'lonely', 'isolated', 
            'can\'t cope', 'overwhelmed', 'giving up'
        ]
        
        # Prohibited medical advice patterns
        self.medical_patterns = [
            r'\b(take|stop taking|start)\s+(medication|medicine|pills|drugs)\b',
            r'\bdiagnos(e|is|ed)\s+with\b',
            r'\bprescri(be|ption)\b',
            r'\b(increase|decrease)\s+dosage\b'
        ]
        
        # Emergency resources
        self.emergency_resources = {
            'US': {
                'suicide': '988 (Suicide & Crisis Lifeline)',
                'general': '911 (Emergency Services)',
                'text': 'Text HOME to 741741 (Crisis Text Line)'
            },
            'International': {
                'suicide': 'Find your country: https://findahelpline.com',
                'general': 'International Association for Suicide Prevention: https://www.iasp.info'
            }
        }
    
    def check_safety(self, message: str) -> Dict:
        """
        Check message for safety concerns
        
        Returns:
            dict: {
                'is_safe': bool,
                'risk_level': str ('none', 'low', 'medium', 'high', 'crisis'),
                'concerns': list,
                'show_resources': bool,
                'resources': dict
            }
        """
        message_lower = message.lower()
        concerns = []
        risk_level = 'none'
        show_resources = False
        
        # Check for crisis keywords (HIGH PRIORITY)
        for category, keywords in self.crisis_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    concerns.append(category)
                    risk_level = 'crisis'
                    show_resources = True
                    break
        
        # Check for warning signs
        if risk_level != 'crisis':
            warning_count = sum(1 for keyword in self.warning_keywords if keyword in message_lower)
            if warning_count >= 3:
                risk_level = 'high'
                show_resources = True
            elif warning_count >= 2:
                risk_level = 'medium'
            elif warning_count >= 1:
                risk_level = 'low'
        
        # Check for medical advice requests
        for pattern in self.medical_patterns:
            if re.search(pattern, message_lower):
                concerns.append('medical_advice_request')
                if risk_level == 'none':
                    risk_level = 'low'
        
        return {
            'is_safe': risk_level not in ['crisis'],
            'risk_level': risk_level,
            'concerns': list(set(concerns)),
            'show_resources': show_resources,
            'resources': self.emergency_resources if show_resources else {}
        }
    
    def get_crisis_response(self, concerns: List[str]) -> str:
        """Generate appropriate crisis response message"""
        
        base_message = "🚨 **I'm concerned about your safety.**\n\n"
        
        if 'suicide' in concerns:
            base_message += (
                "If you're having thoughts of suicide, please reach out for help immediately:\n\n"
                "**🆘 EMERGENCY RESOURCES:**\n"
                "• **Call 988** - Suicide & Crisis Lifeline (US)\n"
                "• **Text HOME to 741741** - Crisis Text Line\n"
                "• **Call 911** - If you're in immediate danger\n"
                "• **International:** https://findahelpline.com\n\n"
                "**You are not alone. Help is available 24/7.**\n\n"
            )
        elif 'self_harm' in concerns:
            base_message += (
                "If you're thinking about hurting yourself, please get help:\n\n"
                "**🆘 SUPPORT RESOURCES:**\n"
                "• **Call 988** - Crisis Support\n"
                "• **Text HOME to 741741** - Crisis Text Line\n"
                "• **Call 911** - If you need immediate help\n\n"
            )
        elif 'violence' in concerns:
            base_message += (
                "If you're having thoughts of hurting others:\n\n"
                "**🆘 GET HELP NOW:**\n"
                "• **Call 911** - Emergency Services\n"
                "• **Call 988** - Crisis Support\n"
                "• Speak with a mental health professional immediately\n\n"
            )
        elif 'abuse' in concerns:
            base_message += (
                "If you're experiencing abuse, help is available:\n\n"
                "**🆘 SUPPORT RESOURCES:**\n"
                "• **National Domestic Violence Hotline:** 1-800-799-7233\n"
                "• **Call 911** - If you're in immediate danger\n"
                "• **Call 988** - Crisis Support\n\n"
            )
        
        base_message += (
            "⚠️ **Please remember:** I'm an AI assistant, not a therapist or crisis counselor. "
            "Professional help is essential in crisis situations."
        )
        
        return base_message
    
    def get_medical_disclaimer(self) -> str:
        """Get medical advice disclaimer"""
        return (
            "⚠️ **Medical Disclaimer:** I cannot provide medical advice, diagnoses, or "
            "treatment recommendations. Please consult with a licensed healthcare professional "
            "for medical concerns, especially regarding medications or mental health diagnoses."
        )
    
    def get_general_disclaimer(self) -> str:
        """Get general disclaimer for the chatbot"""
        return (
            "⚠️ **Important Notice:**\n\n"
            "• This is an AI chatbot, NOT a licensed therapist or medical professional\n"
            "• This tool is for supportive conversations only\n"
            "• NOT a substitute for professional mental health care\n"
            "• In crisis situations, contact emergency services immediately\n"
            "• All conversations are confidential but not therapy sessions"
        )
    
    def check_response_safety(self, response: str) -> Tuple[bool, str]:
        """
        Check if AI-generated response is safe
        
        Returns:
            tuple: (is_safe, reason)
        """
        response_lower = response.lower()
        
        # Check for prohibited content
        prohibited_phrases = [
            'i diagnose you',
            'you have',
            'you should take',
            'stop taking your medication',
            'you don\'t need therapy',
            'you should hurt'
        ]
        
        for phrase in prohibited_phrases:
            if phrase in response_lower:
                return False, f"Response contains prohibited phrase: {phrase}"
        
        return True, "Response is safe"
    
    def log_safety_event(self, user_id: str, risk_level: str, concerns: List[str]):
        """Log safety events for monitoring (implement as needed)"""
        # This can be extended to log to database or file
        print(f"[SAFETY LOG] User: {user_id}, Risk: {risk_level}, Concerns: {concerns}")


# Global instance
safety_monitor = SafetyMonitor()
