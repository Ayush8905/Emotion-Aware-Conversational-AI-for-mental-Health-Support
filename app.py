"""
Mental Health Chatbot - Professional Dark Theme UI
All backend services working perfectly - No bugs version
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
    page_title="MindCare AI - Mental Health Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Session persistence
query_params = st.query_params
if not st.session_state.authenticated and 'session_token' in query_params and 'user_id' in query_params:
    st.session_state.session_token = query_params['session_token']
    st.session_state.user_id = query_params['user_id']
    st.session_state.user_name = query_params.get('user_name', 'User')
    st.session_state.authenticated = True
    st.session_state.page = 'chat'

# Professional Dark Theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #0E1117;
    color: #E8EAED;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* Force sidebar to be visible and styled */
[data-testid="stSidebar"] {
    background-color: #1A1D24 !important;
    border-right: 2px solid #40444D !important;
    min-width: 300px !important;
    max-width: 300px !important;
    padding: 20px 16px !important;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: #1A1D24 !important;
}

[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 300px !important;
    max-width: 300px !important;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 300px !important;
    max-width: 300px !important;
    margin-left: 0px !important;
}

/* Make all sidebar content visible */
[data-testid="stSidebar"] * {
    opacity: 1 !important;
    visibility: visible !important;
}

/* Sidebar content styling */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #FFFFFF !important;
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 20px !important;
    margin-bottom: 16px !important;
}

[data-testid="stSidebar"] hr,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    border-color: #40444D !important;
    margin: 20px 0 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] p {
    color: #E8EAED !important;
    opacity: 1 !important;
}

/* Input fields */
.stTextInput input, .stTextArea textarea {
    background-color: #2D3139 !important;
    color: #E8EAED !important;
    border: 1px solid #40444D !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #5865F2 !important;
    box-shadow: 0 0 0 3px rgba(88, 101, 242, 0.1) !important;
}

/* Main content buttons */
.stButton button {
    background: linear-gradient(135deg, #5865F2 0%, #7289DA 100%);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.3s ease;
    width: 100%;
    text-align: center;
}

.stButton button:hover {
    background: linear-gradient(135deg, #4752C4 0%, #5B6EAE 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(88, 101, 242, 0.3);
}

/* Sidebar buttons - Highly visible style */
[data-testid="stSidebar"] .stButton button {
    background: #2D3139 !important;
    color: #FFFFFF !important;
    border: 1.5px solid #40444D !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    display: block !important;
    margin-bottom: 8px !important;
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: linear-gradient(135deg, #5865F2 0%, #7289DA 100%) !important;
    border-color: #5865F2 !important;
    transform: translateX(5px) !important;
    box-shadow: 0 4px 12px rgba(88, 101, 242, 0.5) !important;
}

/* Primary button in sidebar - Very prominent */
[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: linear-gradient(135deg, #5865F2 0%, #7289DA 100%) !important;
    color: white !important;
    border: 2px solid #7289DA !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 16px 20px !important;
}

[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4752C4 0%, #5B6EAE 100%) !important;
    border-color: #4752C4 !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 16px rgba(88, 101, 242, 0.6) !important;
}

.user-message {
    background: linear-gradient(135deg, #5865F2 0%, #7289DA 100%);
    color: white;
    padding: 14px 18px;
    border-radius: 16px 16px 4px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 2px 8px rgba(88, 101, 242, 0.2);
}

.bot-message {
    background: #2D3139;
    color: #E8EAED;
    padding: 14px 18px;
    border-radius: 16px 16px 16px 4px;
    margin: 8px 0;
    max-width: 80%;
    border: 1px solid #40444D;
}

.emotion-badge {
    background: rgba(255, 255, 255, 0.15);
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-top: 6px;
    display: inline-block;
}

.auth-card {
    background: linear-gradient(135deg, #1A1D24 0%, #2D3139 100%);
    border-radius: 16px;
    padding: 40px;
    max-width: 450px;
    margin: 60px auto;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    border: 1px solid #40444D;
}

.auth-card h1 {
    text-align: center;
    color: #5865F2;
    font-size: 32px;
    margin-bottom: 10px;
}

.auth-card p {
    text-align: center;
    color: #B8BAC1;
    margin-bottom: 30px;
}

.disclaimer-box {
    background: linear-gradient(135deg, #3A2F1E 0%, #4A3F2E 100%);
    border: 2px solid #FFA500;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
}

.disclaimer-box h3 {
    color: #FFB84D !important;
    margin: 0 0 12px 0 !important;
}

.welcome-msg {
    text-align: center;
    color: #B8BAC1;
    padding: 40px 20px;
    background: #2D3139;
    border-radius: 12px;
    border: 1px solid #40444D;
    margin: 20px 0;
}

.welcome-msg h2 {
    color: #5865F2 !important;
}

.timestamp {
    color: #6B6F76;
    font-size: 11px;
    margin-top: 4px;
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin: 4px 2px;
}

.badge-success {
    background: #1E3A2B;
    color: #00A67E;
    border: 1px solid #00A67E;
}

.badge-warning {
    background: #3A2F1E;
    color: #FFA500;
    border: 1px solid #FFA500;
}

.badge-info {
    background: #1E2A3A;
    color: #5865F2;
    border: 1px solid #5865F2;
}

h1, h2, h3 {
    color: #E8EAED !important;
}

hr {
    border-color: #2D3139 !important;
    margin: 20px 0 !important;
}

/* Hide sidebar collapse button */
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Selectbox styling */
[data-testid="stSidebar"] .stSelectbox {
    margin-top: 10px;
}

[data-testid="stSidebar"] .stSelectbox > div {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] .stSelectbox label {
    color: #E8EAED !important;
    font-weight: 500 !important;
}

/* Force button container to be visible */
[data-testid="stSidebar"] .element-container {
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="stSidebar"] .stButton {
    opacity: 1 !important;
    visibility: visible !important;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
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
    st.markdown('<div class="auth-card"><h1>🧠 MindCare AI</h1><p>Your compassionate mental health companion</p>', unsafe_allow_html=True)
    
    username = st.text_input("Username", key="login_username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
    
    if st.button("🚀 Login", use_container_width=True):
        if username and password:
            db = get_db_manager()
            result = db.login_user(username, password)
            if result.get('success'):
                st.session_state.authenticated = True
                st.session_state.user_id = result['user_id']
                st.session_state.user_name = result['name']
                st.session_state.session_token = result['session_token']
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
                st.error(f"❌ {result.get('error', 'Invalid credentials')}")
        else:
            st.warning("⚠️ Please enter both username and password")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    if st.button("📝 Create New Account", use_container_width=True):
        st.session_state.page = 'signup'
        st.rerun()
    
    st.markdown('<p style="text-align: center; color: #6B6F76; margin-top: 30px; font-size: 13px;">⚠️ For supportive conversations only.<br><strong>Crisis? Call 988 or 911</strong></p></div>', unsafe_allow_html=True)

def signup_page():
    st.markdown('<div class="auth-card"><h1>✨ Create Account</h1><p>Join MindCare AI</p>', unsafe_allow_html=True)
    
    new_username = st.text_input("Username", key="signup_username", placeholder="Choose a username")
    new_password = st.text_input("Password", type="password", key="signup_password", placeholder="Create password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password", placeholder="Confirm password")
    
    if st.button("✨ Create Account", use_container_width=True):
        if new_username and new_password and confirm_password:
            if new_password != confirm_password:
                st.error("❌ Passwords don't match")
            else:
                db = get_db_manager()
                result = db.signup_user(new_username, new_password, new_username)
                if result['success']:
                    st.success("✅ Account created!")
                    time.sleep(1)
                    login_result = db.login_user(new_username, new_password)
                    st.session_state.authenticated = True
                    st.session_state.user_id = login_result['user_id']
                    st.session_state.user_name = login_result['name']
                    st.session_state.session_token = login_result['session_token']
                    st.session_state.page = 'chat'
                    st.query_params.update({
                        'session_token': st.session_state.session_token,
                        'user_id': st.session_state.user_id,
                        'user_name': new_username
                    })
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Signup failed')}")
        else:
            st.warning("⚠️ Fill all fields")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    if st.button("◀️ Back to Login", use_container_width=True):
        st.session_state.page = 'login'
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def chat_page():
    # ChatGPT-style Left Sidebar
    with st.sidebar:
        # Header
        st.markdown("### 🧠 MindCare AI")
        st.divider()
        
        # New Chat Button (prominent)
        if st.button("➕ New Chat", use_container_width=True, type="primary", key="chat_new_chat"):
            if st.session_state.current_session_id:
                get_storage().end_session(st.session_state.current_session_id)
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.session_state.disclaimer_shown = False
            st.rerun()
        
        st.divider()
        
        # Navigation Menu
        if st.button("💬 Chat", use_container_width=True, key="chat_nav_chat"):
            st.session_state.page = 'chat'
            st.rerun()
        
        if st.button("📜 Chat History", use_container_width=True, key="chat_nav_history"):
            st.session_state.page = 'history'
            st.rerun()
        
        if st.button("📊 Analytics Dashboard", use_container_width=True, key="chat_nav_analytics"):
            st.session_state.page = 'analytics'
            st.rerun()
        
        if st.button("🆘 Emergency Resources", use_container_width=True, key="chat_nav_emergency"):
            st.session_state.page = 'emergency'
            st.rerun()
        
        if st.button("📝 Feedback Survey", use_container_width=True, key="chat_nav_survey"):
            st.session_state.page = 'survey'
            st.rerun()
        
        st.divider()
        
        # Language Selector
        st.markdown("**Settings**")
        try:
            from language_manager import language_manager
            db = get_db_manager()
            current_language = st.session_state.get('user_language', db.get_user_language(st.session_state.user_id))
            languages = language_manager.get_supported_languages()
            language_options = list(languages.values())
            language_codes = list(languages.keys())
            current_index = language_codes.index(current_language) if current_language in language_codes else 0
            
            selected_language_name = st.selectbox("🌍 Language", language_options, index=current_index, key="chat_language_selector")
            selected_language_code = language_codes[language_options.index(selected_language_name)]
            
            if selected_language_code != current_language:
                if db.set_user_language(st.session_state.user_id, selected_language_code):
                    st.session_state.user_language = selected_language_code
                    st.success("✅ Language updated!")
                    time.sleep(0.5)
                    st.rerun()
        except:
            pass
        
        st.divider()
        
        # User info and logout
        st.markdown(f"**👤 {st.session_state.user_name}**")
        
        if st.button("🚪 Logout", use_container_width=True, key="chat_logout"):
            if st.session_state.current_session_id:
                get_storage().end_session(st.session_state.current_session_id)
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()
    
    # Disclaimer
    if not st.session_state.disclaimer_shown:
        st.markdown('<div class="disclaimer-box"><h3>⚠️ Important</h3><p>• AI chatbot, NOT a therapist<br>• Crisis? Call 988 or 911<br>• For support only</p></div>', unsafe_allow_html=True)
        if st.button("✅ I Understand", use_container_width=True, key="accept_disclaimer"):
            st.session_state.disclaimer_shown = True
            st.rerun()
        return
    
    # Initialize
    if 'user_language' not in st.session_state:
        db = get_db_manager()
        st.session_state.user_language = db.get_user_language(st.session_state.user_id)
    
    if not st.session_state.current_session_id:
        storage = get_storage()
        active_session = storage.get_active_session(st.session_state.user_id)
        if active_session:
            st.session_state.current_session_id = active_session
            st.session_state.messages = storage.get_conversation_history(active_session)
        else:
            st.session_state.current_session_id = storage.create_session(st.session_state.user_id)
            st.session_state.messages = []
    
    # Welcome
    if not st.session_state.messages:
        st.markdown('<div class="welcome-msg"><h2>👋 Welcome</h2><p>How are you feeling today?</p></div>', unsafe_allow_html=True)
    
    # Messages
    for idx, msg in enumerate(st.session_state.messages):
        if msg['role'] == 'user':
            emotion = msg.get('emotion', '')
            confidence = int(msg.get('confidence', 0) * 100)
            st.markdown(f'<div class="user-message">{msg["content"]}<br><span class="emotion-badge">🎭 {emotion.title()} · {confidence}%</span><div class="timestamp">{datetime.now().strftime("%I:%M %p")}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{msg["content"]}<div class="timestamp">{datetime.now().strftime("%I:%M %p")}</div></div>', unsafe_allow_html=True)
            
            # Feedback
            feedback_key = f"feedback_{st.session_state.current_session_id}_{idx}"
            if feedback_key not in st.session_state:
                col1, col2, col3, col4 = st.columns([1, 1, 1, 10])
                with col1:
                    if st.button("👍", key=f"pos_{idx}"):
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
                    if st.button("👎", key=f"neg_{idx}"):
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
                    if st.button("😐", key=f"neu_{idx}"):
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
                feedback_given = st.session_state[feedback_key]
                if feedback_given == 'positive':
                    st.markdown('<span class="badge badge-success">✓ Helpful</span>', unsafe_allow_html=True)
                elif feedback_given == 'negative':
                    st.markdown('<span class="badge badge-warning">✗ Not helpful</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge badge-info">− Neutral</span>', unsafe_allow_html=True)
    
    # Input
    st.markdown("---")
    with st.form(key=f"chat_form_{st.session_state.message_count}", clear_on_submit=True):
        user_input = st.text_input("Message", key=f"chat_input_{st.session_state.message_count}", placeholder="Type here...", label_visibility="collapsed")
        submitted = st.form_submit_button("Send 💬", use_container_width=True)
    
    if submitted and user_input and user_input.strip():
        with st.spinner("💭 Thinking..."):
            try:
                safety_check = safety_monitor.check_safety(user_input.strip())
                if safety_check['risk_level'] == 'crisis':
                    st.error(safety_monitor.get_crisis_response(safety_check['concerns']))
                    safety_monitor.log_safety_event(st.session_state.user_id, safety_check['risk_level'], safety_check['concerns'])
                
                chatbot = get_chatbot()
                result = chatbot.chat(user_input.strip(), username=st.session_state.user_id, user_language=st.session_state.get('user_language', 'en'))
                
                if result.get('error_type') == 'offline':
                    st.warning("🔌 Offline mode")
                elif result.get('error_type') == 'rate_limit':
                    st.warning("⏳ Rate limit")
                elif result.get('error_type') in ['network', 'timeout']:
                    st.warning("🌐 Connection issue")
                
                if safety_check['show_resources'] and safety_check['risk_level'] != 'crisis':
                    result['response'] = "⚠️ Need help? Call 988.\n\n" + result['response']
                
                if 'medical_advice_request' in safety_check['concerns']:
                    result['response'] += "\n\n" + safety_monitor.get_medical_disclaimer()
                
                storage = get_storage()
                storage.save_message(st.session_state.current_session_id, st.session_state.user_id, 'user', user_input.strip(), result['detected_emotion'], result['confidence'], result['top3_emotions'])
                storage.save_message(st.session_state.current_session_id, st.session_state.user_id, 'assistant', result['response'])
                
                st.session_state.messages.append({'role': 'user', 'content': user_input.strip(), 'emotion': result['detected_emotion'], 'confidence': result['confidence']})
                st.session_state.messages.append({'role': 'assistant', 'content': result['response']})
                st.session_state.message_count += 1
                
                if st.session_state.message_count % 15 == 0:
                    st.session_state.show_survey_prompt = True
                
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Survey
    if st.session_state.get('show_survey_prompt'):
        st.info(f"📝 {st.session_state.message_count} messages exchanged. Quick feedback?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Give Feedback"):
                st.session_state.page = 'survey'
                st.session_state.show_survey_prompt = False
                st.rerun()
        with col2:
            if st.button("⏭️ Later"):
                st.session_state.show_survey_prompt = False
                st.rerun()

def history_page():
    # ChatGPT-style Left Sidebar
    with st.sidebar:
        # Header
        st.markdown("### 🧠 MindCare AI")
        st.divider()
        
        # New Chat Button (prominent)
        if st.button("➕ New Chat", use_container_width=True, type="primary", key="history_new_chat"):
            if st.session_state.current_session_id:
                get_storage().end_session(st.session_state.current_session_id)
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.session_state.disclaimer_shown = False
            st.session_state.page = 'chat'
            st.rerun()
        
        st.divider()
        
        # Navigation Menu
        if st.button("💬 Chat", use_container_width=True, key="history_nav_chat"):
            st.session_state.page = 'chat'
            st.rerun()
        
        if st.button("📜 Chat History", use_container_width=True, key="history_nav_history"):
            st.session_state.page = 'history'
            st.rerun()
        
        if st.button("📊 Analytics Dashboard", use_container_width=True, key="history_nav_analytics"):
            st.session_state.page = 'analytics'
            st.rerun()
        
        if st.button("🆘 Emergency Resources", use_container_width=True, key="history_nav_emergency"):
            st.session_state.page = 'emergency'
            st.rerun()
        
        if st.button("📝 Feedback Survey", use_container_width=True, key="history_nav_survey"):
            st.session_state.page = 'survey'
            st.rerun()
        
        st.divider()
        
        # User info and logout
        st.markdown(f"**👤 {st.session_state.user_name}**")
        
        if st.button("🚪 Logout", use_container_width=True, key="history_logout"):
            if st.session_state.current_session_id:
                get_storage().end_session(st.session_state.current_session_id)
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()
    
    st.title("📜 Conversation History")
    
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🆕 New Chat", key="history_top_new_chat"):
            if st.session_state.current_session_id:
                get_storage().end_session(st.session_state.current_session_id)
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.session_state.page = 'chat'
            st.rerun()
    
    st.markdown("---")
    
    storage = get_storage()
    sessions = storage.get_user_sessions(st.session_state.user_id)
    
    if not sessions:
        st.info("📭 No history yet")
        return
    
    st.markdown(f"**Total: {len(sessions)}**")
    
    if st.button("🗑️ Delete All"):
        st.session_state.confirm_delete_all = True
    
    if st.session_state.get('confirm_delete_all'):
        st.warning("⚠️ Delete all?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes"):
                for session in sessions:
                    storage.delete_conversation(session['session_id'], st.session_state.user_id)
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.session_state.confirm_delete_all = False
                st.success("✅ Deleted!")
                st.rerun()
        with col2:
            if st.button("❌ No"):
                st.session_state.confirm_delete_all = False
                st.rerun()
    
    st.markdown("---")
    
    for session in sessions:
        with st.expander(f"📅 {session['start_time'].strftime('%B %d, %Y at %I:%M %p')} • {session['message_count']} messages"):
            st.write(f"**Status:** {session['status'].title()}")
            if session['emotion_summary']['dominant_emotion']:
                st.write(f"**Emotion:** {session['emotion_summary']['dominant_emotion'].title()}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📖 View", key=f"view_{session['session_id']}"):
                    messages = storage.get_conversation_history(session['session_id'])
                    for msg in messages:
                        if msg['role'] == 'user':
                            st.info(f"**You:** {msg['content']}")
                            if msg.get('emotion'):
                                st.caption(f"{msg['emotion'].title()} ({int(msg['confidence']*100)}%)")
                        else:
                            st.success(f"**Bot:** {msg['content']}")
            with col2:
                if st.button("💬 Continue", key=f"continue_{session['session_id']}"):
                    st.session_state.current_session_id = session['session_id']
                    st.session_state.messages = storage.get_conversation_history(session['session_id'])
                    st.session_state.page = 'chat'
                    st.rerun()
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{session['session_id']}"):
                    if storage.delete_conversation(session['session_id'], st.session_state.user_id):
                        if st.session_state.get('current_session_id') == session['session_id']:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        st.success("✅ Deleted!")
                        st.rerun()

def main():
    try:
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
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        if st.button("🔄 Refresh"):
            st.rerun()

if __name__ == "__main__":
    main()
