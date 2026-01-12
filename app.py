"""
Mental Health Chatbot - Professional Web UI
"""

import streamlit as st
import time
from datetime import datetime
import uuid
from database_manager import DatabaseManager
from conversation_storage import ConversationStorage
from chatbot_pipeline import MentalHealthChatbot
from safety_monitor import safety_monitor
from emergency_page import show_emergency_page
from feedback_system import feedback_system
from analytics_dashboard import show_analytics_dashboard
from satisfaction_survey import show_satisfaction_survey
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'session_token' not in st.session_state:
    st.session_state.session_token = None
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'message_count' not in st.session_state:
    st.session_state.message_count = 0
if 'disclaimer_shown' not in st.session_state:
    st.session_state.disclaimer_shown = False
if 'terms_accepted' not in st.session_state:
    st.session_state.terms_accepted = False

# Session persistence via query params
query_params = st.query_params
if not st.session_state.authenticated and 'session_token' in query_params and 'user_id' in query_params:
    st.session_state.session_token = query_params['session_token']
    st.session_state.user_id = query_params['user_id']
    st.session_state.user_name = query_params.get('user_name', 'User')
    st.session_state.authenticated = True
    st.session_state.page = 'chat'

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
#MainMenu, footer { visibility: hidden; }
.stTextInput input, .stTextArea textarea {
    background-color: white !important;
    color: #333 !important;
    border-radius: 10px;
    border: 2px solid #e0e0e0;
    padding: 12px;
}
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    padding: 12px;
    font-weight: 600;
}
.user-message {
    background: #667eea;
    color: white;
    padding: 15px 20px;
    border-radius: 20px 20px 5px 20px;
    margin: 10px 0;
}
.bot-message {
    background: #f5f5f5;
    color: #333;
    padding: 15px 20px;
    border-radius: 20px 20px 20px 5px;
    margin: 10px 0;
}
.emotion-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 12px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db_manager():
    return DatabaseManager()

@st.cache_resource
def get_storage():
    return ConversationStorage()

@st.cache_resource
def get_chatbot():
    return MentalHealthChatbot()

def login_page():
    st.markdown('<div style="background: white; padding: 40px; border-radius: 20px; max-width: 450px; margin: 50px auto;">', unsafe_allow_html=True)
    st.title("🧠 Mental Health Chatbot")
    st.markdown("### Welcome Back!")
    
    # Safety disclaimer
    st.warning("⚠️ **Important:** This is an AI assistant, NOT a licensed therapist. In crisis situations, call 988 (US) or emergency services.")
    st.markdown("---")
    
    email = st.text_input("📧 Email", key="login_email")
    password = st.text_input("🔒 Password", type="password", key="login_password")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Login", use_container_width=True):
            if email and password:
                with st.spinner("Logging in..."):
                    db = get_db_manager()
                    result = db.login_user(email, password)
                    
                    if result['success']:
                        st.session_state.authenticated = True
                        st.session_state.user_id = result['user_id']
                        st.session_state.user_name = result['name']
                        st.session_state.session_token = result.get('session_token', str(uuid.uuid4()))
                        st.session_state.page = 'chat'
                        
                        st.query_params.update({
                            'session_token': st.session_state.session_token,
                            'user_id': st.session_state.user_id,
                            'user_name': st.session_state.user_name
                        })
                        
                        st.success("✅ Login successful!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
            else:
                st.warning("⚠️ Please fill all fields")
    
    with col2:
        if st.button("📝 Sign Up", use_container_width=True):
            st.session_state.page = 'signup'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def signup_page():
    st.markdown('<div style="background: white; padding: 40px; border-radius: 20px; max-width: 450px; margin: 50px auto;">', unsafe_allow_html=True)
    st.title("🧠 Mental Health Chatbot")
    st.markdown("### Create Account")
    st.markdown("---")
    
    name = st.text_input("👤 Full Name", key="signup_name")
    email = st.text_input("📧 Email", key="signup_email")
    password = st.text_input("🔒 Password", type="password", key="signup_password")
    confirm = st.text_input("🔒 Confirm Password", type="password", key="signup_confirm")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ Create Account", use_container_width=True):
            if name and email and password and confirm:
                if password != confirm:
                    st.error("❌ Passwords don't match!")
                elif len(password) < 6:
                    st.error("❌ Password must be 6+ characters")
                else:
                    with st.spinner("Creating account..."):
                        db = get_db_manager()
                        result = db.signup_user(email, password, name)
                        
                        if result['success']:
                            st.success("✅ Account created! Please login.")
                            time.sleep(1)
                            st.session_state.page = 'login'
                            st.rerun()
                        else:
                            st.error(f"❌ {result['error']}")
            else:
                st.warning("⚠️ Please fill all fields")
    
    with col2:
        if st.button("◀️ Back", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def chat_page():
    # Show disclaimer on first use
    if not st.session_state.disclaimer_shown:
        st.markdown(
            '<div style="background: #fff3cd; border: 2px solid #ffc107; padding: 20px; '
            'border-radius: 10px; margin-bottom: 20px;">'
            '<h3 style="color: #856404; margin: 0;">⚠️ Important Safety Notice</h3>'
            '<p style="color: #856404; margin: 10px 0;">'
            '• This is an <strong>AI chatbot</strong>, NOT a licensed therapist or medical professional<br>'
            '• For supportive conversations only - NOT a substitute for professional care<br>'
            '• <strong>In crisis? Call 988 (Suicide & Crisis Lifeline) or 911 immediately</strong><br>'
            '• All conversations are confidential but not therapy sessions'
            '</p>'
            '</div>',
            unsafe_allow_html=True
        )
        if st.button("✅ I Understand", key="accept_disclaimer"):
            st.session_state.disclaimer_shown = True
            st.rerun()
        return
    
    col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
    with col1:
        st.title(f"💬 Welcome {st.session_state.user_name}!")
    with col2:
        if st.button("🆘 Emergency", help="Crisis resources and helplines"):
            st.session_state.page = 'emergency'
            st.rerun()
    with col3:
        if st.button("📊 Analytics", help="View feedback and statistics"):
            st.session_state.page = 'analytics'
            st.rerun()
    with col4:
        if st.button("🆕 New Chat"):
            # End current session
            if st.session_state.current_session_id:
                get_storage().end_session(st.session_state.current_session_id)
            
            # Clear messages and session
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.rerun()
    with col5:
        if st.button("📜 History"):
            st.session_state.page = 'history'
            st.rerun()
    with col6:
        if st.button("🚪 Logout"):
            if st.session_state.current_session_id:
                get_storage().end_session(st.session_state.current_session_id)
            
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Language Selector in Sidebar
    with st.sidebar:
        st.markdown("### 🌍 Language / Idioma / Langue")
        
        # Get language manager
        try:
            from language_manager import language_manager
            db = get_db_manager()
            
            # Get current user language
            current_language = st.session_state.get('user_language', db.get_user_language(st.session_state.user_id))
            
            # Language selector
            languages = language_manager.get_supported_languages()
            language_options = list(languages.values())
            language_codes = list(languages.keys())
            
            current_index = language_codes.index(current_language) if current_language in language_codes else 0
            
            selected_language_name = st.selectbox(
                "Select Language:",
                language_options,
                index=current_index,
                key="language_selector"
            )
            
            # Get selected language code
            selected_language_code = language_codes[language_options.index(selected_language_name)]
            
            # If language changed, update database and session
            if selected_language_code != current_language:
                if db.set_user_language(st.session_state.user_id, selected_language_code):
                    st.session_state.user_language = selected_language_code
                    st.success("✅ Language updated!")
                    st.rerun()
        except ImportError:
            st.warning("Language support not available")
        
        st.markdown("---")
        st.markdown("**Current User:** " + st.session_state.user_name)
    
    # Initialize session_state for user language if not set
    if 'user_language' not in st.session_state:
        db = get_db_manager()
        st.session_state.user_language = db.get_user_language(st.session_state.user_id)
    
    st.markdown("---")
    
    # Initialize session
    if not st.session_state.current_session_id:
        storage = get_storage()
        active_session = storage.get_active_session(st.session_state.user_id)
        if active_session:
            st.session_state.current_session_id = active_session
            st.session_state.messages = storage.get_conversation_history(active_session)
        else:
            st.session_state.current_session_id = storage.create_session(st.session_state.user_id)
            st.session_state.messages = []
    
    # Display messages
    if not st.session_state.messages:
        st.info("👋 Hi! I'm here to listen. How are you feeling today?")
    
    for idx, msg in enumerate(st.session_state.messages):
        if msg['role'] == 'user':
            emotion = msg.get('emotion', '')
            confidence = int(msg.get('confidence', 0) * 100)
            st.markdown(
                f'<div class="user-message">{msg["content"]}<br>'
                f'<span class="emotion-badge">{emotion.title()} {confidence}%</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
            
            # Add feedback buttons for bot responses
            feedback_key = f"feedback_{st.session_state.current_session_id}_{idx}"
            
            # Check if feedback already given
            if feedback_key not in st.session_state:
                col1, col2, col3, col4 = st.columns([1, 1, 1, 10])
                
                with col1:
                    if st.button("👍", key=f"pos_{idx}", help="Helpful response"):
                        # Get corresponding user message
                        user_msg_idx = idx - 1
                        user_msg = st.session_state.messages[user_msg_idx] if user_msg_idx >= 0 else {'content': '', 'emotion': 'unknown'}
                        
                        feedback_system.record_response_feedback(
                            username=st.session_state.user_id,
                            conversation_id=st.session_state.current_session_id,
                            message_index=idx,
                            user_message=user_msg.get('content', ''),
                            bot_response=msg['content'],
                            detected_emotion=user_msg.get('emotion', 'unknown'),
                            rating='positive'
                        )
                        st.session_state[feedback_key] = 'positive'
                        st.rerun()
                
                with col2:
                    if st.button("👎", key=f"neg_{idx}", help="Not helpful"):
                        user_msg_idx = idx - 1
                        user_msg = st.session_state.messages[user_msg_idx] if user_msg_idx >= 0 else {'content': '', 'emotion': 'unknown'}
                        
                        feedback_system.record_response_feedback(
                            username=st.session_state.user_id,
                            conversation_id=st.session_state.current_session_id,
                            message_index=idx,
                            user_message=user_msg.get('content', ''),
                            bot_response=msg['content'],
                            detected_emotion=user_msg.get('emotion', 'unknown'),
                            rating='negative'
                        )
                        st.session_state[feedback_key] = 'negative'
                        st.rerun()
                
                with col3:
                    if st.button("😐", key=f"neu_{idx}", help="Neutral"):
                        user_msg_idx = idx - 1
                        user_msg = st.session_state.messages[user_msg_idx] if user_msg_idx >= 0 else {'content': '', 'emotion': 'unknown'}
                        
                        feedback_system.record_response_feedback(
                            username=st.session_state.user_id,
                            conversation_id=st.session_state.current_session_id,
                            message_index=idx,
                            user_message=user_msg.get('content', ''),
                            bot_response=msg['content'],
                            detected_emotion=user_msg.get('emotion', 'unknown'),
                            rating='neutral'
                        )
                        st.session_state[feedback_key] = 'neutral'
                        st.rerun()
            else:
                # Show feedback was given
                feedback_given = st.session_state[feedback_key]
                if feedback_given == 'positive':
                    st.markdown("✅ *Marked as helpful*")
                elif feedback_given == 'negative':
                    st.markdown("❌ *Marked as not helpful*")
                else:
                    st.markdown("😐 *Marked as neutral*")
    
    # Input form
    st.markdown("---")
    with st.form(key=f"chat_form_{st.session_state.message_count}", clear_on_submit=True):
        user_input = st.text_input(
            "Message",
            key=f"chat_input_{st.session_state.message_count}",
            placeholder="Type your message and press Enter...",
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("Send 📤", use_container_width=True)
    
    if submitted and user_input and user_input.strip():
        with st.spinner("💭 Thinking..."):
            try:
                # Safety check on user input
                safety_check = safety_monitor.check_safety(user_input.strip())
                
                # If crisis detected, show emergency resources immediately
                if safety_check['risk_level'] == 'crisis':
                    crisis_response = safety_monitor.get_crisis_response(safety_check['concerns'])
                    st.error(crisis_response)
                    
                    # Log safety event
                    safety_monitor.log_safety_event(
                        st.session_state.user_id,
                        safety_check['risk_level'],
                        safety_check['concerns']
                    )
                
                chatbot = get_chatbot()
                user_language = st.session_state.get('user_language', 'en')
                result = chatbot.chat(user_input.strip(), username=st.session_state.user_id, user_language=user_language)
                
                # Check for offline mode or errors
                if result.get('fallback_used') or result.get('error_type'):
                    if result.get('error_type') == 'offline':
                        st.warning("🔌 **Offline Mode**: You're currently offline. Limited functionality available.")
                    elif result.get('error_type') == 'rate_limit':
                        st.warning("⏳ **Rate Limit**: Too many requests. Please wait a moment.")
                    elif result.get('error_type') in ['network', 'timeout']:
                        st.warning("🌐 **Connection Issue**: Having trouble connecting. Using fallback response.")
                    elif result.get('fallback_used'):
                        st.info("ℹ️ **Fallback Response**: Using pre-configured response due to service issues.")
                
                # Add safety warning if needed
                if safety_check['show_resources'] and safety_check['risk_level'] != 'crisis':
                    result['response'] = (
                        "⚠️ I notice you might be going through a difficult time. "
                        "Please remember: **Call 988 for crisis support** or **911 for emergencies**. "
                        "Professional help is available 24/7.\n\n" + result['response']
                    )
                
                # Check for medical advice requests
                if 'medical_advice_request' in safety_check['concerns']:
                    result['response'] += (
                        "\n\n" + safety_monitor.get_medical_disclaimer()
                    )
                
                storage = get_storage()
                storage.save_message(
                    st.session_state.current_session_id,
                    st.session_state.user_id,
                    'user',
                    user_input.strip(),
                    result['detected_emotion'],
                    result['confidence'],
                    result['top3_emotions']
                )
                storage.save_message(
                    st.session_state.current_session_id,
                    st.session_state.user_id,
                    'assistant',
                    result['response']
                )
                
                st.session_state.messages.append({
                    'role': 'user',
                    'content': user_input.strip(),
                    'emotion': result['detected_emotion'],
                    'confidence': result['confidence']
                })
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result['response']
                })
                
                st.session_state.message_count += 1
                
                # Show survey prompt after every 15 messages
                if st.session_state.message_count > 0 and st.session_state.message_count % 15 == 0:
                    st.session_state.show_survey_prompt = True
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Survey prompt if triggered
    if 'show_survey_prompt' in st.session_state and st.session_state.show_survey_prompt:
        st.markdown("---")
        st.info(f"📝 You've exchanged {st.session_state.message_count} messages. Would you like to share quick feedback?")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📝 Give Feedback", key="survey_yes"):
                st.session_state.page = 'survey'
                st.session_state.show_survey_prompt = False
                st.rerun()
        
        with col2:
            if st.button("⏭️ Later", key="survey_skip"):
                st.session_state.show_survey_prompt = False
                st.rerun()
    
    elif submitted and not user_input.strip():
        st.warning("⚠️ Please type a message!")

def history_page():
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("📜 History")
    with col2:
        if st.button("◀️ Back"):
            st.session_state.page = 'chat'
            st.rerun()
    
    st.markdown("---")
    
    storage = get_storage()
    sessions = storage.get_user_sessions(st.session_state.user_id)
    
    if not sessions:
        st.info("No history yet. Start chatting!")
        return
    
    # Add Delete All button
    st.markdown(f"**Total Conversations:** {len(sessions)}")
    col_del1, col_del2, col_del3 = st.columns([2, 2, 6])
    with col_del1:
        if st.button("🗑️ Delete All History", type="secondary"):
            st.session_state.confirm_delete_all = True
    
    # Confirmation dialog for delete all
    if st.session_state.get('confirm_delete_all', False):
        with col_del2:
            st.warning("⚠️ Are you sure?")
        with col_del3:
            col_yes, col_no, _ = st.columns([1, 1, 8])
            with col_yes:
                if st.button("✅ Yes, Delete All"):
                    deleted_count = 0
                    for session in sessions:
                        if storage.delete_conversation(session['session_id'], st.session_state.user_id):
                            deleted_count += 1
                    st.session_state.current_session_id = None
                    st.session_state.messages = []
                    st.session_state.confirm_delete_all = False
                    st.success(f"✅ Deleted {deleted_count} conversations!")
                    st.rerun()
            with col_no:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_delete_all = False
                    st.rerun()
    
    st.markdown("---")
    
    for session in sessions:
        with st.expander(f"📅 {session['start_time'].strftime('%B %d, %Y at %I:%M %p')} ({session['message_count']} messages)"):
            st.write(f"**Status:** {session['status'].title()}")
            
            if session['emotion_summary']['dominant_emotion']:
                st.write(f"**Main Emotion:** {session['emotion_summary']['dominant_emotion'].title()}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📖 View", key=f"view_{session['session_id']}"):
                    messages = storage.get_conversation_history(session['session_id'])
                    st.markdown("### Conversation:")
                    for msg in messages:
                        if msg['role'] == 'user':
                            st.info(f"**You:** {msg['content']}")
                            if msg.get('emotion'):
                                st.caption(f"Emotion: {msg['emotion'].title()} ({int(msg['confidence']*100)}%)")
                        else:
                            st.success(f"**Bot:** {msg['content']}")
            
            with col2:
                if st.button(f"💬 Continue", key=f"continue_{session['session_id']}"):
                    st.session_state.current_session_id = session['session_id']
                    st.session_state.messages = storage.get_conversation_history(session['session_id'])
                    st.session_state.message_count = 0
                    st.session_state.page = 'chat'
                    st.rerun()
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{session['session_id']}", type="secondary"):
                    if storage.delete_conversation(session['session_id'], st.session_state.user_id):
                        st.success("✅ Conversation deleted successfully!")
                        # If this was the active session, clear it
                        if st.session_state.get('current_session_id') == session['session_id']:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete conversation.")

def main():
    if not st.session_state.authenticated:
        if st.session_state.page == 'signup':
            signup_page()
        else:
            login_page()
    else:
        if st.session_state.page == 'history':
            history_page()
        elif st.session_state.page == 'emergency':
            show_emergency_page()
        elif st.session_state.page == 'analytics':
            show_analytics_dashboard()
        elif st.session_state.page == 'survey':
            survey_completed = show_satisfaction_survey(
                conversation_id=st.session_state.current_session_id,
                username=st.session_state.user_id,
                message_count=st.session_state.message_count
            )
            if survey_completed:
                st.session_state.page = 'chat'
                st.rerun()
        else:
            chat_page()

if __name__ == "__main__":
    main()
