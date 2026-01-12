"""
Satisfaction Survey for Phase 7 User Study
Post-conversation user satisfaction assessment
"""

import streamlit as st
from feedback_system import feedback_system

def show_satisfaction_survey(conversation_id: str, username: str, message_count: int):
    """
    Display post-conversation satisfaction survey
    
    Args:
        conversation_id: ID of the completed conversation
        username: Username of the user
        message_count: Number of messages exchanged
    """
    
    st.markdown("## 📝 Help Us Improve!")
    st.markdown("### Please share your feedback about this conversation")
    
    st.info(f"💬 You exchanged {message_count} messages in this conversation")
    
    with st.form(key="satisfaction_survey"):
        st.markdown("### Rate Your Experience")
        
        # Overall satisfaction
        overall = st.slider(
            "Overall Satisfaction",
            min_value=1,
            max_value=5,
            value=3,
            help="How satisfied are you with this conversation?"
        )
        st.markdown("*1 = Very Dissatisfied, 5 = Very Satisfied*")
        
        st.markdown("---")
        
        # Empathy rating
        empathy = st.slider(
            "Empathy & Understanding",
            min_value=1,
            max_value=5,
            value=3,
            help="Did the chatbot understand your feelings?"
        )
        st.markdown("*1 = Not at all, 5 = Completely*")
        
        st.markdown("---")
        
        # Helpfulness rating
        helpfulness = st.slider(
            "Helpfulness of Responses",
            min_value=1,
            max_value=5,
            value=3,
            help="Were the responses helpful and relevant?"
        )
        st.markdown("*1 = Not helpful, 5 = Very helpful*")
        
        st.markdown("---")
        
        # Ease of use
        ease_of_use = st.slider(
            "Ease of Use",
            min_value=1,
            max_value=5,
            value=3,
            help="Was the chatbot easy to use?"
        )
        st.markdown("*1 = Very difficult, 5 = Very easy*")
        
        st.markdown("---")
        
        # Would recommend
        would_recommend = st.radio(
            "Would you recommend this chatbot to others?",
            options=["Yes", "No", "Maybe"],
            horizontal=True
        )
        
        st.markdown("---")
        
        # Comments
        comments = st.text_area(
            "What did you like most about this conversation?",
            placeholder="Share what worked well...",
            height=100
        )
        
        # Suggestions
        suggestions = st.text_area(
            "What could be improved?",
            placeholder="Share your suggestions for improvement...",
            height=100
        )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            submit_button = st.form_submit_button(
                "✅ Submit Feedback",
                use_container_width=True,
                type="primary"
            )
        
        with col2:
            skip_button = st.form_submit_button(
                "⏭️ Skip",
                use_container_width=True
            )
        
        if submit_button:
            # Record survey
            would_recommend_bool = True if would_recommend == "Yes" else False if would_recommend == "No" else None
            
            success = feedback_system.record_satisfaction_survey(
                username=username,
                conversation_id=conversation_id,
                overall_satisfaction=overall,
                empathy_rating=empathy,
                helpfulness_rating=helpfulness,
                ease_of_use=ease_of_use,
                would_recommend=would_recommend_bool,
                comments=comments if comments else None,
                suggestions=suggestions if suggestions else None
            )
            
            if success:
                st.success("✅ Thank you for your feedback! Your input helps us improve.")
                st.balloons()
                return True
            else:
                st.error("❌ Failed to save feedback. Please try again.")
                return False
        
        if skip_button:
            st.info("⏭️ Survey skipped. You can provide feedback later from the Analytics page.")
            return True
    
    return False


def show_inline_survey_prompt(message_count: int) -> bool:
    """
    Show a small prompt to take the survey after certain message count
    
    Args:
        message_count: Number of messages in current conversation
    
    Returns:
        True if user wants to take survey, False otherwise
    """
    # Show prompt after 10, 20, 30 messages etc.
    if message_count > 0 and message_count % 10 == 0:
        st.info("📝 You've exchanged {} messages. Would you like to share quick feedback?".format(message_count))
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📝 Give Feedback", key=f"survey_prompt_{message_count}"):
                return True
        
        with col2:
            if st.button("⏭️ Later", key=f"survey_skip_{message_count}"):
                return False
    
    return False


def show_quick_feedback_form():
    """Show a quick 1-question feedback form"""
    
    st.markdown("### Quick Feedback")
    
    with st.form(key="quick_feedback"):
        rating = st.radio(
            "How is the conversation going so far?",
            options=["😊 Great", "😐 Okay", "😟 Not well"],
            horizontal=True
        )
        
        comment = st.text_input(
            "Any quick thoughts? (optional)",
            placeholder="Brief comment..."
        )
        
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            st.success("✅ Thanks for the quick feedback!")
            return True
    
    return False


# Test function
if __name__ == "__main__":
    st.set_page_config(page_title="Satisfaction Survey Test", page_icon="📝")
    
    # Mock session state
    if 'username' not in st.session_state:
        st.session_state.username = "test_user"
    
    show_satisfaction_survey(
        conversation_id="test_conv_123",
        username=st.session_state.username,
        message_count=15
    )
