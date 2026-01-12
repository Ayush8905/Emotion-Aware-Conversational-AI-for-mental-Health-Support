"""
Mental Health Chatbot - Professional Web UI
Fully Animated, Responsive, Accessible Design
"""

import streamlit as st
import time
from datetime import datetime
import uuid
from database_manager import DatabaseManager
from conversation_storage import ConversationStorage
from chatbot_pipeline import MentalHealthChatbot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with Animations
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Smooth Transitions */
    .stButton button, .stTextInput input, .stTextArea textarea {
        transition: all 0.3s ease;
    }
    
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass Morphism Container */
    .glass-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Animated Heading */
    .animated-heading {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #fff, #e0e0e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
        animation: slideDown 0.8s ease-out;
    }
    
    @keyframes slideDown {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Button Styles with Hover Effects */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 4px 15px 0 rgba(116, 79, 168, 0.75);
        position: relative;
        overflow: hidden;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(116, 79, 168, 0.95);
    }
    
    .stButton button::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    /* Input Field Styles */
    .stTextInput input, .stTextArea textarea {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 15px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        transform: scale(1.02);
    }
    
    /* Chat Message Styles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: auto;
        max-width: 70%;
        animation: slideInRight 0.4s ease-out;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .bot-message {
        background: rgba(255, 255, 255, 0.95);
        color: #333;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: auto;
        max-width: 70%;
        animation: slideInLeft 0.4s ease-out;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Emotion Badge with Pulse Animation */
    .emotion-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 5px;
        animation: pulse 2s infinite;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .emotion-joy { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .emotion-sadness { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .emotion-anger { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }
    .emotion-fear { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; }
    .emotion-neutral { background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%); color: #333; }
    
    /* Loading Spinner */
    .loader {
        border: 5px solid rgba(255, 255, 255, 0.3);
        border-top: 5px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        gap: 5px;
        padding: 20px;
        animation: fadeIn 0.3s ease-in;
    }
    
    .typing-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #667eea;
        animation: typingDot 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typingDot {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.7; }
        30% { transform: translateY(-10px); opacity: 1; }
    }
    
    /* History Card */
    .history-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .history-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    
    /* Confidence Progress Bar */
    .confidence-bar {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        animation: fillBar 1s ease-out;
    }
    
    @keyframes fillBar {
        from { width: 0%; }
    }
    
    /* Success/Error Messages */
    .success-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        animation: slideDown 0.5s ease-out;
    }
    
    .error-message {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        animation: shake 0.5s ease-out;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .animated-heading { font-size: 2rem; }
        .user-message, .bot-message { max-width: 90%; }
        .glass-container { padding: 20px; }
    }
    
    /* Smooth Page Transitions */
    .page-transition {
        animation: pageSlide 0.5s ease-out;
    }
    
    @keyframes pageSlide {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Floating Animation for Icons */
    .floating {
        animation: floating 3s ease-in-out infinite;
    }
    
    @keyframes floating {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Glow Effect */
    .glow {
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.8); }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'conversation_storage' not in st.session_state:
        st.session_state.conversation_storage = ConversationStorage()

# Login Page
def login_page():
    st.markdown('<div class="page-transition">', unsafe_allow_html=True)
    st.markdown('<h1 class="animated-heading floating">🧠 Mental Health Chatbot</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="glass-container glow">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            email = st.text_input("Email", key="login_email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            
            if st.button("🚀 Login", key="login_btn"):
                with st.spinner(""):
                    st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
                    time.sleep(0.5)
                    
                    result = st.session_state.db_manager.login_user(email, password)
                    
                    if result['success']:
                        st.session_state.logged_in = True
                        st.session_state.user_id = result['user_id']
                        st.session_state.user_name = result['name']
                        st.session_state.page = 'chat'
                        st.markdown(f'<div class="success-message">✅ {result["message"]}</div>', unsafe_allow_html=True)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.markdown(f'<div class="error-message">❌ {result["error"]}</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Create Account")
            name = st.text_input("Full Name", key="signup_name", placeholder="Enter your name")
            email = st.text_input("Email", key="signup_email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", key="signup_password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password", placeholder="Confirm your password")
            
            if st.button("✨ Create Account", key="signup_btn"):
                if password != confirm_password:
                    st.markdown('<div class="error-message">❌ Passwords do not match!</div>', unsafe_allow_html=True)
                else:
                    with st.spinner(""):
                        st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
                        time.sleep(0.5)
                        
                        result = st.session_state.db_manager.signup_user(email, password, name)
                        
                        if result['success']:
                            st.markdown(f'<div class="success-message">✅ {result["message"]} Please login.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="error-message">❌ {result["error"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat Page
def chat_page():
    st.markdown('<div class="page-transition">', unsafe_allow_html=True)
    
    # Header with navigation
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        st.markdown(f'<h3 style="color: white;">👋 Hi, {st.session_state.user_name}!</h3>', unsafe_allow_html=True)
    with col2:
        st.markdown('<h2 class="animated-heading" style="font-size: 2rem;">💬 Chat with AI Therapist</h2>', unsafe_allow_html=True)
    with col3:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("📜 History"):
                st.session_state.page = 'history'
                st.rerun()
        with col_b:
            if st.button("🔄 New Chat"):
                st.session_state.session_id = None
                st.session_state.messages = []
                st.session_state.chatbot = None
                st.rerun()
        with col_c:
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.page = 'login'
                st.session_state.messages = []
                st.rerun()
    
    st.markdown("---")
    
    # Initialize chatbot and session
    if st.session_state.chatbot is None:
        with st.spinner(""):
            st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
            st.session_state.chatbot = MentalHealthChatbot()
    
    if st.session_state.session_id is None:
        st.session_state.session_id = st.session_state.conversation_storage.create_session(
            st.session_state.user_id
        )
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for idx, message in enumerate(st.session_state.messages):
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Bot message with emotion
                emotion_html = ""
                if 'emotion' in message:
                    emotion = message['emotion']
                    emotion_class = f"emotion-{emotion.lower()}" if emotion.lower() in ['joy', 'sadness', 'anger', 'fear', 'neutral'] else 'emotion-neutral'
                    emotion_html = f'<span class="emotion-badge {emotion_class}">{emotion}</span>'
                    
                    if 'confidence' in message:
                        confidence_pct = int(message['confidence'] * 100)
                        emotion_html += f"""
                        <div style="margin-top: 10px;">
                            <small style="color: #666;">Confidence: {confidence_pct}%</small>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence_pct}%;"></div>
                            </div>
                        </div>
                        """
                
                st.markdown(f"""
                <div class="bot-message">
                    {message['content']}
                    <div style="margin-top: 10px;">{emotion_html}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.text_area(
        "Your message",
        placeholder="Type your message here... Share what's on your mind.",
        height=100,
        key="user_input"
    )
    
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        send_button = st.button("💌 Send", key="send_btn", use_container_width=True)
    
    if send_button and user_input.strip():
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # Show typing indicator
        with st.spinner(""):
            st.markdown("""
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get bot response
            response = st.session_state.chatbot.chat(user_input)
            
            # Add bot message
            bot_message = {
                'role': 'assistant',
                'content': response['response']
            }
            
            if 'emotion' in response:
                bot_message['emotion'] = response['emotion']
            if 'confidence' in response:
                bot_message['confidence'] = response['confidence']
            
            st.session_state.messages.append(bot_message)
            
            # Save to database
            st.session_state.conversation_storage.save_message(
                session_id=st.session_state.session_id,
                message=user_input,
                sender='user'
            )
            
            st.session_state.conversation_storage.save_message(
                session_id=st.session_state.session_id,
                message=response['response'],
                sender='assistant',
                detected_emotion=response.get('emotion'),
                emotion_confidence=response.get('confidence')
            )
        
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# History Page
def history_page():
    st.markdown('<div class="page-transition">', unsafe_allow_html=True)
    
    # Header
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        st.markdown(f'<h3 style="color: white;">👋 Hi, {st.session_state.user_name}!</h3>', unsafe_allow_html=True)
    with col2:
        st.markdown('<h2 class="animated-heading" style="font-size: 2rem;">📜 Conversation History</h2>', unsafe_allow_html=True)
    with col3:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("💬 Back to Chat"):
                st.session_state.page = 'chat'
                st.rerun()
        with col_b:
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.page = 'login'
                st.rerun()
    
    st.markdown("---")
    
    # Get user sessions
    sessions = st.session_state.conversation_storage.get_user_sessions(st.session_state.user_id)
    
    if not sessions:
        st.markdown("""
        <div class="glass-container" style="text-align: center; padding: 50px;">
            <h3 style="color: white;">No conversations yet</h3>
            <p style="color: white;">Start chatting to see your history here!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for session in sessions:
            session_id = str(session['_id'])
            start_time = session.get('start_time', 'Unknown')
            message_count = session.get('message_count', 0)
            
            # Get emotion summary
            emotions = session.get('emotion_summary', {})
            emotion_badges = ""
            for emotion, count in list(emotions.items())[:3]:  # Top 3 emotions
                emotion_class = f"emotion-{emotion.lower()}" if emotion.lower() in ['joy', 'sadness', 'anger', 'fear', 'neutral'] else 'emotion-neutral'
                emotion_badges += f'<span class="emotion-badge {emotion_class}">{emotion}: {count}</span>'
            
            st.markdown(f"""
            <div class="history-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0;">💬 Conversation</h4>
                        <small style="color: #666;">Started: {start_time}</small><br>
                        <small style="color: #666;">Messages: {message_count}</small>
                    </div>
                    <div>
                        {emotion_badges}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Buttons for each session
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                if st.button(f"📖 View Conversation", key=f"view_{session_id}"):
                    show_conversation_details(session_id)
            with col2:
                if st.button(f"▶️ Continue", key=f"continue_{session_id}"):
                    st.session_state.session_id = session_id
                    st.session_state.messages = load_session_messages(session_id)
                    st.session_state.page = 'chat'
                    st.rerun()
            with col3:
                if st.button(f"📥 Export", key=f"export_{session_id}"):
                    export_conversation(session_id)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Helper function to show conversation details
def show_conversation_details(session_id):
    messages = st.session_state.conversation_storage.get_conversation_history(session_id)
    
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white;">📖 Conversation Details</h3>', unsafe_allow_html=True)
    
    for msg in messages:
        sender = msg.get('sender', 'unknown')
        content = msg.get('message', '')
        
        if sender == 'user':
            st.markdown(f"""
            <div class="user-message" style="max-width: 100%;">
                {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            emotion = msg.get('detected_emotion', '')
            emotion_html = ""
            if emotion:
                emotion_class = f"emotion-{emotion.lower()}" if emotion.lower() in ['joy', 'sadness', 'anger', 'fear', 'neutral'] else 'emotion-neutral'
                emotion_html = f'<span class="emotion-badge {emotion_class}">{emotion}</span>'
            
            st.markdown(f"""
            <div class="bot-message" style="max-width: 100%;">
                {content}
                <div style="margin-top: 10px;">{emotion_html}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Helper function to load session messages
def load_session_messages(session_id):
    messages = st.session_state.conversation_storage.get_conversation_history(session_id)
    loaded_messages = []
    
    for msg in messages:
        sender = msg.get('sender', 'unknown')
        content = msg.get('message', '')
        
        message_obj = {
            'role': 'user' if sender == 'user' else 'assistant',
            'content': content
        }
        
        if sender == 'assistant':
            if 'detected_emotion' in msg:
                message_obj['emotion'] = msg['detected_emotion']
            if 'emotion_confidence' in msg:
                message_obj['confidence'] = msg['emotion_confidence']
        
        loaded_messages.append(message_obj)
    
    return loaded_messages

# Helper function to export conversation
def export_conversation(session_id):
    result = st.session_state.conversation_storage.export_conversation(session_id)
    
    if result:
        st.markdown(f"""
        <div class="success-message">
            ✅ Conversation exported successfully!<br>
            <small>File: {result}</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-message">
            ❌ Failed to export conversation
        </div>
        """, unsafe_allow_html=True)

# Main function
def main():
    load_custom_css()
    init_session_state()
    
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.page == 'chat':
            chat_page()
        elif st.session_state.page == 'history':
            history_page()

if __name__ == "__main__":
    main()
