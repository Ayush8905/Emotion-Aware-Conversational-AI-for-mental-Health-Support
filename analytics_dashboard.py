"""
Analytics Dashboard for User Study & Validation (Phase 7)
Visualizes feedback data and system performance metrics
"""

import streamlit as st
from feedback_system import feedback_system
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

def show_analytics_dashboard():
    """Display comprehensive analytics dashboard"""
    
    st.title("📊 Analytics Dashboard")
    st.markdown("### User Study & Validation Metrics")
    
    # Check if user is logged in
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        st.warning("⚠️ Please login to view analytics")
        return
    
    # Only show full analytics to admin users (you can add admin check here)
    # For now, showing all users their own stats plus overall stats
    
    st.markdown("---")
    
    # Get statistics
    stats = feedback_system.get_feedback_statistics()
    
    if not stats:
        st.info("📊 No feedback data available yet. Start chatting and rate responses!")
        return
    
    # Overview Metrics
    st.markdown("## 📈 Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Feedback",
            value=stats.get('total_feedback', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Satisfaction Rate",
            value=f"{stats.get('satisfaction_rate', 0)}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Total Surveys",
            value=stats.get('total_surveys', 0),
            delta=None
        )
    
    with col4:
        recommend_rate = stats.get('recommend_rate', 0)
        st.metric(
            label="Would Recommend",
            value=f"{recommend_rate}%",
            delta=None
        )
    
    st.markdown("---")
    
    # Feedback Distribution
    st.markdown("## 👍 Feedback Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for feedback ratings
        if stats.get('total_feedback', 0) > 0:
            feedback_data = {
                'Rating': ['Positive', 'Negative', 'Neutral'],
                'Count': [
                    stats.get('positive', 0),
                    stats.get('negative', 0),
                    stats.get('neutral', 0)
                ]
            }
            df_feedback = pd.DataFrame(feedback_data)
            
            fig = px.pie(
                df_feedback,
                values='Count',
                names='Rating',
                title='Response Ratings',
                color='Rating',
                color_discrete_map={
                    'Positive': '#28a745',
                    'Negative': '#dc3545',
                    'Neutral': '#ffc107'
                }
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No feedback data yet")
    
    with col2:
        # Bar chart for survey ratings
        if stats.get('total_surveys', 0) > 0:
            survey_data = {
                'Metric': ['Overall', 'Empathy', 'Helpfulness', 'Ease of Use'],
                'Average Rating': [
                    stats.get('avg_satisfaction', 0),
                    stats.get('avg_empathy', 0),
                    stats.get('avg_helpfulness', 0),
                    stats.get('avg_ease_of_use', 0)
                ]
            }
            df_survey = pd.DataFrame(survey_data)
            
            fig = px.bar(
                df_survey,
                x='Metric',
                y='Average Rating',
                title='Average Survey Ratings (1-5 scale)',
                color='Average Rating',
                color_continuous_scale='Viridis',
                range_y=[0, 5]
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No survey data yet")
    
    st.markdown("---")
    
    # Emotion-based Feedback
    st.markdown("## 😊 Feedback by Detected Emotion")
    
    emotion_breakdown = feedback_system.get_emotion_feedback_breakdown()
    
    if emotion_breakdown:
        # Prepare data for visualization
        emotions = []
        positive_counts = []
        negative_counts = []
        neutral_counts = []
        satisfaction_rates = []
        
        for emotion, data in emotion_breakdown.items():
            emotions.append(emotion)
            positive_counts.append(data['positive'])
            negative_counts.append(data['negative'])
            neutral_counts.append(data['neutral'])
            satisfaction_rates.append(data['satisfaction_rate'])
        
        # Create stacked bar chart
        fig = go.Figure(data=[
            go.Bar(name='Positive', x=emotions, y=positive_counts, marker_color='#28a745'),
            go.Bar(name='Negative', x=emotions, y=negative_counts, marker_color='#dc3545'),
            go.Bar(name='Neutral', x=emotions, y=neutral_counts, marker_color='#ffc107')
        ])
        
        fig.update_layout(
            barmode='stack',
            title='Feedback Distribution by Emotion',
            xaxis_title='Detected Emotion',
            yaxis_title='Feedback Count',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Show emotion satisfaction table
        st.markdown("### Emotion Satisfaction Rates")
        
        emotion_df = pd.DataFrame([
            {
                'Emotion': emotion,
                'Total Feedback': data['total'],
                'Positive': data['positive'],
                'Negative': data['negative'],
                'Satisfaction %': f"{data['satisfaction_rate']}%"
            }
            for emotion, data in emotion_breakdown.items()
        ])
        
        st.dataframe(emotion_df, width='stretch')
    else:
        st.info("No emotion feedback data yet")
    
    st.markdown("---")
    
    # Recent Feedback
    st.markdown("## 💬 Recent Feedback")
    
    recent_feedback = feedback_system.get_recent_feedback(limit=10)
    
    if recent_feedback:
        for i, feedback in enumerate(recent_feedback, 1):
            with st.expander(f"Feedback #{i} - {feedback.get('rating', 'N/A').title()} ({feedback.get('detected_emotion', 'unknown')})"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.write(f"**User:** {feedback.get('username', 'Anonymous')}")
                    st.write(f"**Rating:** {feedback.get('rating', 'N/A').title()}")
                    st.write(f"**Emotion:** {feedback.get('detected_emotion', 'Unknown')}")
                    
                    timestamp = feedback.get('timestamp')
                    if isinstance(timestamp, str):
                        st.write(f"**Time:** {timestamp[:19]}")
                    else:
                        st.write(f"**Time:** {timestamp}")
                
                with col2:
                    st.write("**User Message:**")
                    st.info(feedback.get('user_message', 'N/A'))
                    
                    st.write("**Bot Response:**")
                    st.success(feedback.get('bot_response', 'N/A')[:200] + "...")
                    
                    if feedback.get('comment'):
                        st.write("**User Comment:**")
                        st.warning(feedback.get('comment'))
    else:
        st.info("No recent feedback available")
    
    st.markdown("---")
    
    # Performance Metrics
    st.markdown("## ⚡ Performance Metrics")
    
    avg_response_time = feedback_system.get_average_response_time()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Avg Response Time",
            value=f"{avg_response_time:.2f}s" if avg_response_time > 0 else "N/A",
            delta=None
        )
    
    with col2:
        # Calculate uptime (mock - would need actual monitoring)
        st.metric(
            label="System Uptime",
            value="99.5%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Total Users",
            value="N/A",  # Would need user count from database
            delta=None
        )
    
    st.markdown("---")
    
    # Export Options
    st.markdown("## 📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Feedback Data", key="export_feedback"):
            feedback_data = feedback_system.export_feedback_data()
            if feedback_data:
                df = pd.DataFrame(feedback_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                st.success(f"✅ Exported {len(feedback_data)} feedback records")
            else:
                st.warning("No data to export")
    
    with col2:
        if st.button("📋 Generate Report", key="generate_report"):
            st.info("Report generation feature coming soon...")
    
    with col3:
        if st.button("🔄 Refresh Data", key="refresh_data"):
            st.rerun()
    
    st.markdown("---")
    
    # User's Personal Stats
    st.markdown("## 👤 Your Feedback History")
    
    user_feedback = feedback_system.get_user_feedback_history(
        st.session_state.user_id,
        limit=20
    )
    
    if user_feedback:
        st.write(f"You have provided **{len(user_feedback)}** feedback responses")
        
        # Count user's ratings
        user_positive = sum(1 for f in user_feedback if f.get('rating') == 'positive')
        user_negative = sum(1 for f in user_feedback if f.get('rating') == 'negative')
        user_neutral = sum(1 for f in user_feedback if f.get('rating') == 'neutral')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👍 Positive", user_positive)
        
        with col2:
            st.metric("👎 Negative", user_negative)
        
        with col3:
            st.metric("😐 Neutral", user_neutral)
        
        # Show detailed history
        with st.expander("View Detailed History"):
            for feedback in user_feedback[:10]:
                st.markdown(f"""
                **Conversation:** {feedback.get('conversation_id', 'N/A')}  
                **Rating:** {feedback.get('rating', 'N/A').title()}  
                **Emotion:** {feedback.get('detected_emotion', 'unknown')}  
                **Time:** {feedback.get('timestamp')}
                ---
                """)
    else:
        st.info("You haven't provided any feedback yet. Rate chatbot responses to help us improve!")
    
    # Back button
    st.markdown("---")
    if st.button("◀️ Back to Chat", key="back_to_chat", use_container_width=True):
        st.session_state.page = 'chat'
        st.rerun()


if __name__ == "__main__":
    # For testing
    st.set_page_config(
        page_title="Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Mock session state for testing
    if 'username' not in st.session_state:
        st.session_state.username = "test_user"
    
    show_analytics_dashboard()
