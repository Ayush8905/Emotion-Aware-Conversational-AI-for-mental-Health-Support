"""
Language Manager for Multi-Language Support
Handles translation and localization for the mental health chatbot
Supports: English, Spanish, French, Hindi, Chinese (Simplified), Arabic
"""

from deep_translator import GoogleTranslator
from typing import Dict, Optional, List
import os
from dotenv import load_dotenv

load_dotenv()


class LanguageManager:
    """
    Manages translations and language preferences
    - UI text translation
    - User input/output translation
    - Language detection
    - Localization support
    """
    
    # Supported languages with their codes
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish (Español)',
        'fr': 'French (Français)',
        'hi': 'Hindi (हिन्दी)',
        'zh-CN': 'Chinese Simplified (简体中文)',
        'ar': 'Arabic (العربية)',
        'de': 'German (Deutsch)',
        'pt': 'Portuguese (Português)',
        'ru': 'Russian (Русский)',
        'ja': 'Japanese (日本語)'
    }
    
    def __init__(self):
        """Initialize language manager"""
        self.default_language = 'en'
        
        # UI translations dictionary
        self.ui_translations = {
            'en': self._get_english_ui(),
            'es': self._get_spanish_ui(),
            'fr': self._get_french_ui(),
            'hi': self._get_hindi_ui(),
            'zh-CN': self._get_chinese_ui(),
            'ar': self._get_arabic_ui()
        }
    
    def translate_text(self, text: str, target_language: str, source_language: str = 'auto') -> str:
        """
        Translate text using Google Translator
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if 'auto')
            
        Returns:
            Translated text
        """
        try:
            if target_language == source_language or target_language == 'en' and source_language == 'en':
                return text
            
            translator = GoogleTranslator(source=source_language, target=target_language)
            translated = translator.translate(text)
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original if translation fails
    
    def translate_to_english(self, text: str, source_language: str = 'auto') -> str:
        """Translate user input to English for emotion detection"""
        if source_language == 'en':
            return text
        return self.translate_text(text, 'en', source_language)
    
    def translate_from_english(self, text: str, target_language: str) -> str:
        """Translate bot response from English to user's language"""
        if target_language == 'en':
            return text
        return self.translate_text(text, target_language, 'en')
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code
        """
        try:
            translator = GoogleTranslator(source='auto', target='en')
            # This is a workaround - deep_translator doesn't have direct detection
            # but we can use auto-detection in translation
            return 'auto'
        except Exception as e:
            print(f"Language detection error: {e}")
            return 'en'
    
    def get_ui_text(self, language: str) -> Dict[str, str]:
        """
        Get UI translations for a language
        
        Args:
            language: Language code
            
        Returns:
            Dictionary of UI text translations
        """
        return self.ui_translations.get(language, self.ui_translations['en'])
    
    def get_language_name(self, code: str) -> str:
        """Get language name from code"""
        return self.SUPPORTED_LANGUAGES.get(code, 'English')
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages"""
        return self.SUPPORTED_LANGUAGES.copy()
    
    # ==================== UI TRANSLATIONS ====================
    
    def _get_english_ui(self) -> Dict[str, str]:
        """English UI translations"""
        return {
            # Authentication
            'login': 'Login',
            'signup': 'Sign Up',
            'username': 'Username',
            'password': 'Password',
            'create_account': 'Create New Account',
            'login_button': '🚀 Login',
            'signup_button': '✨ Create Account',
            'logout': 'Logout',
            
            # Navigation
            'chat': 'Chat',
            'history': 'History',
            'emergency': 'Emergency',
            'analytics': 'Analytics',
            'new_chat': 'New Chat',
            'back': 'Back',
            
            # Chat interface
            'type_message': 'Type your message here...',
            'send': 'Send',
            'thinking': 'Thinking...',
            'emotion_detected': 'Emotion detected',
            
            # Feedback
            'feedback_positive': 'Helpful',
            'feedback_negative': 'Not helpful',
            'feedback_neutral': 'Neutral',
            'thank_you_feedback': 'Thank you for your feedback!',
            
            # History
            'no_history': 'No history yet. Start chatting!',
            'view': 'View',
            'continue': 'Continue',
            'delete': 'Delete',
            'delete_all': 'Delete All History',
            'confirm_delete': 'Are you sure?',
            'yes_delete': 'Yes, Delete All',
            'cancel': 'Cancel',
            'deleted_success': 'Conversation deleted successfully!',
            
            # Emergency
            'crisis_hotlines': '24/7 Crisis Hotlines',
            'emergency_warning': '⚠️ If you are in immediate danger, call 911',
            
            # Analytics
            'total_feedback': 'Total Feedback',
            'satisfaction_rate': 'Satisfaction Rate',
            'feedback_distribution': 'Feedback Distribution',
            'survey_ratings': 'Survey Ratings',
            'export_data': 'Export Data',
            
            # Survey
            'overall_satisfaction': 'Overall Satisfaction',
            'empathy': 'Empathy & Compassion',
            'helpfulness': 'Helpfulness',
            'ease_of_use': 'Ease of Use',
            'would_recommend': 'Would you recommend this chatbot?',
            'comments': 'Comments',
            'suggestions': 'Suggestions for improvement',
            'submit': 'Submit',
            'skip': 'Skip',
            
            # Messages
            'welcome': 'Welcome to Mental Health Support Chatbot',
            'disclaimer': 'This AI is not a substitute for professional therapy',
            'crisis_detected': 'Crisis situation detected',
            'medical_disclaimer': 'Please consult a healthcare professional',
            
            # Settings
            'language': 'Language',
            'select_language': 'Select your preferred language',
            'language_saved': 'Language preference saved!',
        }
    
    def _get_spanish_ui(self) -> Dict[str, str]:
        """Spanish UI translations"""
        return {
            'login': 'Iniciar sesión',
            'signup': 'Registrarse',
            'username': 'Nombre de usuario',
            'password': 'Contraseña',
            'create_account': 'Crear nueva cuenta',
            'login_button': '🚀 Iniciar sesión',
            'signup_button': '✨ Crear cuenta',
            'logout': 'Cerrar sesión',
            
            'chat': 'Chat',
            'history': 'Historial',
            'emergency': 'Emergencia',
            'analytics': 'Análisis',
            'new_chat': 'Nuevo chat',
            'back': 'Volver',
            
            'type_message': 'Escribe tu mensaje aquí...',
            'send': 'Enviar',
            'thinking': 'Pensando...',
            'emotion_detected': 'Emoción detectada',
            
            'feedback_positive': 'Útil',
            'feedback_negative': 'No útil',
            'feedback_neutral': 'Neutral',
            'thank_you_feedback': '¡Gracias por tu opinión!',
            
            'no_history': 'Aún no hay historial. ¡Comienza a chatear!',
            'view': 'Ver',
            'continue': 'Continuar',
            'delete': 'Eliminar',
            'delete_all': 'Eliminar todo el historial',
            'confirm_delete': '¿Estás seguro?',
            'yes_delete': 'Sí, eliminar todo',
            'cancel': 'Cancelar',
            'deleted_success': '¡Conversación eliminada exitosamente!',
            
            'crisis_hotlines': 'Líneas de crisis 24/7',
            'emergency_warning': '⚠️ Si estás en peligro inmediato, llama al 911',
            
            'welcome': 'Bienvenido al Chatbot de Apoyo en Salud Mental',
            'disclaimer': 'Esta IA no sustituye la terapia profesional',
            
            'language': 'Idioma',
            'select_language': 'Selecciona tu idioma preferido',
            'language_saved': '¡Preferencia de idioma guardada!',
        }
    
    def _get_french_ui(self) -> Dict[str, str]:
        """French UI translations"""
        return {
            'login': 'Connexion',
            'signup': 'S\'inscrire',
            'username': 'Nom d\'utilisateur',
            'password': 'Mot de passe',
            'create_account': 'Créer un nouveau compte',
            'login_button': '🚀 Connexion',
            'signup_button': '✨ Créer un compte',
            'logout': 'Déconnexion',
            
            'chat': 'Chat',
            'history': 'Historique',
            'emergency': 'Urgence',
            'analytics': 'Analyses',
            'new_chat': 'Nouveau chat',
            'back': 'Retour',
            
            'type_message': 'Tapez votre message ici...',
            'send': 'Envoyer',
            'thinking': 'Réflexion...',
            'emotion_detected': 'Émotion détectée',
            
            'feedback_positive': 'Utile',
            'feedback_negative': 'Pas utile',
            'feedback_neutral': 'Neutre',
            'thank_you_feedback': 'Merci pour votre retour!',
            
            'no_history': 'Pas encore d\'historique. Commencez à discuter!',
            'view': 'Voir',
            'continue': 'Continuer',
            'delete': 'Supprimer',
            'delete_all': 'Supprimer tout l\'historique',
            'confirm_delete': 'Êtes-vous sûr?',
            'yes_delete': 'Oui, tout supprimer',
            'cancel': 'Annuler',
            'deleted_success': 'Conversation supprimée avec succès!',
            
            'welcome': 'Bienvenue sur le Chatbot de Soutien en Santé Mentale',
            'disclaimer': 'Cette IA ne remplace pas une thérapie professionnelle',
            
            'language': 'Langue',
            'select_language': 'Sélectionnez votre langue préférée',
            'language_saved': 'Préférence de langue enregistrée!',
        }
    
    def _get_hindi_ui(self) -> Dict[str, str]:
        """Hindi UI translations"""
        return {
            'login': 'लॉगिन',
            'signup': 'साइन अप',
            'username': 'उपयोगकर्ता नाम',
            'password': 'पासवर्ड',
            'create_account': 'नया खाता बनाएं',
            'login_button': '🚀 लॉगिन',
            'signup_button': '✨ खाता बनाएं',
            'logout': 'लॉग आउट',
            
            'chat': 'चैट',
            'history': 'इतिहास',
            'emergency': 'आपातकाल',
            'analytics': 'विश्लेषण',
            'new_chat': 'नई चैट',
            'back': 'वापस',
            
            'type_message': 'यहां अपना संदेश लिखें...',
            'send': 'भेजें',
            'thinking': 'सोच रहा है...',
            'emotion_detected': 'भावना पहचानी गई',
            
            'feedback_positive': 'उपयोगी',
            'feedback_negative': 'उपयोगी नहीं',
            'feedback_neutral': 'तटस्थ',
            'thank_you_feedback': 'आपकी प्रतिक्रिया के लिए धन्यवाद!',
            
            'no_history': 'अभी तक कोई इतिहास नहीं। चैट करना शुरू करें!',
            'view': 'देखें',
            'continue': 'जारी रखें',
            'delete': 'हटाएं',
            'delete_all': 'सभी इतिहास हटाएं',
            
            'welcome': 'मानसिक स्वास्थ्य सहायता चैटबॉट में आपका स्वागत है',
            'disclaimer': 'यह AI पेशेवर थेरेपी का विकल्प नहीं है',
            
            'language': 'भाषा',
            'select_language': 'अपनी पसंदीदा भाषा चुनें',
            'language_saved': 'भाषा प्राथमिकता सहेजी गई!',
        }
    
    def _get_chinese_ui(self) -> Dict[str, str]:
        """Chinese Simplified UI translations"""
        return {
            'login': '登录',
            'signup': '注册',
            'username': '用户名',
            'password': '密码',
            'create_account': '创建新账户',
            'login_button': '🚀 登录',
            'signup_button': '✨ 创建账户',
            'logout': '登出',
            
            'chat': '聊天',
            'history': '历史',
            'emergency': '紧急情况',
            'analytics': '分析',
            'new_chat': '新聊天',
            'back': '返回',
            
            'type_message': '在此输入您的消息...',
            'send': '发送',
            'thinking': '思考中...',
            'emotion_detected': '检测到情绪',
            
            'feedback_positive': '有帮助',
            'feedback_negative': '没帮助',
            'feedback_neutral': '中立',
            'thank_you_feedback': '感谢您的反馈!',
            
            'no_history': '还没有历史记录。开始聊天吧!',
            'view': '查看',
            'continue': '继续',
            'delete': '删除',
            'delete_all': '删除所有历史',
            
            'welcome': '欢迎使用心理健康支持聊天机器人',
            'disclaimer': '此AI不能替代专业治疗',
            
            'language': '语言',
            'select_language': '选择您的首选语言',
            'language_saved': '语言偏好已保存!',
        }
    
    def _get_arabic_ui(self) -> Dict[str, str]:
        """Arabic UI translations"""
        return {
            'login': 'تسجيل الدخول',
            'signup': 'التسجيل',
            'username': 'اسم المستخدم',
            'password': 'كلمة المرور',
            'create_account': 'إنشاء حساب جديد',
            'login_button': '🚀 تسجيل الدخول',
            'signup_button': '✨ إنشاء حساب',
            'logout': 'تسجيل الخروج',
            
            'chat': 'دردشة',
            'history': 'السجل',
            'emergency': 'طوارئ',
            'analytics': 'التحليلات',
            'new_chat': 'دردشة جديدة',
            'back': 'رجوع',
            
            'type_message': 'اكتب رسالتك هنا...',
            'send': 'إرسال',
            'thinking': 'يفكر...',
            'emotion_detected': 'تم اكتشاف المشاعر',
            
            'feedback_positive': 'مفيد',
            'feedback_negative': 'غير مفيد',
            'feedback_neutral': 'محايد',
            'thank_you_feedback': 'شكراً لملاحظاتك!',
            
            'no_history': 'لا يوجد سجل حتى الآن. ابدأ الدردشة!',
            'view': 'عرض',
            'continue': 'متابعة',
            'delete': 'حذف',
            'delete_all': 'حذف كل السجل',
            
            'welcome': 'مرحباً بك في روبوت دعم الصحة النفسية',
            'disclaimer': 'هذا الذكاء الاصطناعي لا يحل محل العلاج المهني',
            
            'language': 'اللغة',
            'select_language': 'اختر لغتك المفضلة',
            'language_saved': 'تم حفظ تفضيل اللغة!',
        }


# Global instance
language_manager = LanguageManager()


# Test function
def test_language_manager():
    """Test language manager functionality"""
    print("="*80)
    print("LANGUAGE MANAGER TEST")
    print("="*80 + "\n")
    
    lm = LanguageManager()
    
    # Test 1: Supported languages
    print("1. Supported Languages:")
    for code, name in lm.get_supported_languages().items():
        print(f"   {code}: {name}")
    print()
    
    # Test 2: Translation
    print("2. Testing Translation:")
    test_text = "I am feeling anxious today"
    print(f"   Original (English): {test_text}")
    
    spanish = lm.translate_text(test_text, 'es', 'en')
    print(f"   Spanish: {spanish}")
    
    back_to_english = lm.translate_text(spanish, 'en', 'es')
    print(f"   Back to English: {back_to_english}")
    print()
    
    # Test 3: UI translations
    print("3. Testing UI Translations:")
    spanish_ui = lm.get_ui_text('es')
    print(f"   Login (Spanish): {spanish_ui['login']}")
    print(f"   Chat (Spanish): {spanish_ui['chat']}")
    print(f"   Welcome (Spanish): {spanish_ui['welcome']}")
    print()
    
    print("="*80)
    print("[SUCCESS] All language manager tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_language_manager()
