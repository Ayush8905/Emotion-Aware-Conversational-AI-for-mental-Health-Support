"""
Emergency Resources Page
Provides crisis hotlines and mental health resources
"""

import streamlit as st

def show_emergency_page():
    """Display emergency resources and crisis helplines"""
    
    st.error("### 🆘 Emergency Resources - Help is Available 24/7")
    
    # Back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("◀️ Back to Chat", key="back_top_btn", use_container_width=True):
            st.session_state.page = 'chat'
            st.rerun()
    
    st.markdown("---")
    
    # Crisis Hotlines Section
    st.markdown("## 🚨 Immediate Crisis Support")
    
    st.warning("""
    ### If you're in immediate danger:
    # 📞 Call 911 (USA)
    Emergency services for life-threatening situations
    """)
    
    st.markdown("### 🆘 Crisis & Suicide Prevention")
    
    # 988 Lifeline
    with st.container():
        st.info("""
        #### 988 Suicide & Crisis Lifeline (USA)
        # 📞 Call or Text: 988
        Free, confidential support 24/7 for people in distress
        
        🔗 Website: [988lifeline.org](https://988lifeline.org)
        """)
    
    # Crisis Text Line
    with st.container():
        st.info("""
        #### Crisis Text Line
        # 💬 Text: HOME to 741741
        24/7 crisis support via text message
        
        🔗 Website: [crisistextline.org](https://www.crisistextline.org)
        """)
    
    # Veterans Crisis Line
    with st.container():
        st.info("""
        #### Veterans Crisis Line
        # 📞 Call: 988 then Press 1
        💬 Text: 838255
        
        Specialized support for veterans and their families
        """)
    
    st.markdown("---")
    
    # Mental Health Support
    st.markdown("## 💚 Mental Health Support")
    
    with st.container():
        st.success("""
        #### SAMHSA National Helpline
        # 📞 1-800-662-4357 (HELP)
        Treatment referral and information service for mental health and substance use
        
        24/7, 365 days a year | Free & Confidential
        """)
    
    with st.container():
        st.success("""
        #### National Domestic Violence Hotline
        # 📞 1-800-799-7233 (SAFE)
        💬 Text: START to 88788
        
        Support for domestic violence victims
        """)
    
    with st.container():
        st.success("""
        #### RAINN National Sexual Assault Hotline
        # 📞 1-800-656-4673 (HOPE)
        Support for sexual assault survivors
        """)
    
    with st.container():
        st.success("""
        #### Trevor Project (LGBTQ+ Youth)
        # 📞 1-866-488-7386
        💬 Text: START to 678678
        
        Crisis support for LGBTQ+ young people
        """)
    
    st.markdown("---")
    
    # International Resources
    st.markdown("## 🌍 International Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        #### Find A Helpline
        Comprehensive directory of crisis helplines worldwide
        
        🔗 [findahelpline.com](https://findahelpline.com)
        """)
    
    with col2:
        st.info("""
        #### International Association for Suicide Prevention
        Crisis centers and helplines by country
        
        🔗 [IASP Resources](https://www.iasp.info/resources/Crisis_Centres)
        """)
    
    st.markdown("---")
    
    # Remember section
    st.info("""
    ### ℹ️ Remember
    
    ✅ You are not alone - help is available
    
    ✅ These services are free and confidential
    
    ✅ Trained counselors are available 24/7
    
    ✅ Reaching out for help is a sign of strength
    
    ✅ Your life matters
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Back button at bottom
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("◀️ Return to Chat", key="back_bottom_btn", use_container_width=True):
            st.session_state.page = 'chat'
            st.rerun()
