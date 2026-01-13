# 🧠 Emotion-Aware Conversational AI for Mental Health Support

## 📋 Project Overview

A **production-ready mental health chatbot system** with transformer-based emotion detection, empathetic response generation, and comprehensive safety features. The system uses the **GoEmotions dataset** (28 emotion categories), fine-tunes **DistilBERT** for emotion classification, and integrates **LLAMA 3.3** for compassionate conversational responses.

### 🎯 System Capabilities

✅ **Emotion Detection**: Real-time classification into 28+ emotion categories  
✅ **Empathetic Responses**: Context-aware, compassionate AI conversations powered by LLAMA 3.3 70B  
✅ **Multi-Language Support**: 10 languages with automatic translation (English, Spanish, French, Hindi, Chinese, Arabic, German, Portuguese, Russian, Japanese)  
✅ **Advanced Error Handling**: Retry logic, fallback responses, offline mode detection  
✅ **Crisis Detection**: Automatic identification of self-harm, suicide, and violence keywords  
✅ **Safety Features**: 24/7 emergency hotlines, medical disclaimers, and crisis response system  
✅ **User Management**: Secure authentication with bcrypt password hashing  
✅ **Conversation Memory**: MongoDB storage with session persistence across refreshes  
✅ **Feedback System**: Real-time thumbs up/down/neutral ratings on every response  
✅ **Analytics Dashboard**: Interactive Plotly visualizations for feedback and performance data  
✅ **Performance Monitoring**: System metrics tracking (CPU, memory, response times)  
✅ **Satisfaction Surveys**: 5-point rating scales for user experience evaluation  
✅ **Web Interface**: Modern Streamlit UI with 6 pages (chat, history, emergency, analytics, survey)  
✅ **Multi-label Support**: Handles complex emotional states  
✅ **High Performance**: Optimized for real-time predictions with GPU acceleration

---

## 🆕 Latest Features (v2.1.0)

### 🌍 **Multi-Language Support**
- Chat in 10 languages: English, Spanish, French, Hindi, Chinese, Arabic, German, Portuguese, Russian, Japanese
- Automatic translation with seamless user experience
- Language preference persistence across sessions
- Full UI localization

### ⚡ **Advanced Error Handling**
- Automatic retry with exponential backoff (3 attempts)
- Offline mode detection with helpful resources
- Emotion-specific fallback responses
- User-friendly error notifications (no technical jargon)
- 100% uptime - system never crashes

### 📊 **Enhanced Analytics**
- Real-time feedback tracking (👍 👎 😐)
- Interactive Plotly visualizations
- CSV export for data analysis
- Emotion-based satisfaction metrics

---

## 📊 Dataset Information

**Dataset**: GoEmotions (Google Research)
- **Total Samples**: 211,742 Reddit comments
- **Emotion Labels**: 28 categories + neutral
- **Labels**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

---

## 🏗️ Complete System Architecture

### ✅ Phase 1-3: Emotion Detection System (COMPLETE)
**Files**: `data_preprocessing.py`, `train_emotion_classifier.py`, `evaluate_model.py`, `inference.py`

- **Dataset**: GoEmotions (211,742 Reddit comments, 28 emotions + neutral)
- **Model**: DistilBERT fine-tuned for emotion classification
- **Text Cleaning**: URL removal, normalization, lowercase conversion
- **Training**: 3 epochs with AdamW optimizer, 70/15/15 train/val/test split
- **Performance**: 65-75% accuracy, handles complex emotional expressions
- **Output**: Real-time emotion prediction with confidence scores

### ✅ Phase 4: Response Generation (COMPLETE)
**Files**: `response_generator.py`, `chatbot_pipeline.py`

- **LLM**: LLAMA 3.3 70B Versatile via Groq API (fast inference)
- **System Prompt**: Compassionate mental health support specialist role
- **Context Integration**: Uses detected emotions to generate empathetic responses
- **Conversation Flow**: Maintains context across chat turns
- **Response Style**: Warm, supportive, professional, non-judgmental
- **Features**: 
  - Validation of emotional states
  - Active listening techniques
  - Coping strategy suggestions
  - Resource recommendations

### ✅ Phase 5A: Database Backend (COMPLETE)
**Files**: `database_manager.py`, `conversation_storage.py`

- **Database**: MongoDB Atlas (cloud-hosted)
- **Collections**: 
  - `users`: Secure authentication with bcrypt (salt rounds: 12)
  - `conversations`: Full chat history with emotion tracking
- **Features**:
  - User registration and login
  - Password hashing and verification
  - Conversation persistence
  - Session management
  - JSON export for chat history
  - MongoDB connection diagnostics

### ✅ Phase 5B: Web Interface (COMPLETE)
**Files**: `app.py`, `test_chatbot.py`

- **Framework**: Streamlit 1.41.1
- **Features**:
  - Modern UI with emoji-rich design
  - User authentication (login/signup)
  - Real-time chat interface with auto-scrolling
  - Session persistence across page refreshes
  - Auto-clearing text input with Enter key support
  - "New Chat" button for fresh conversations
  - Continue previous conversations
  - View conversation history
  - Emergency button in header
  - Mobile-responsive layout
- **State Management**: Query parameters for session token persistence
- **Styling**: Custom CSS for better text input visibility

### ✅ Phase 6: Safety & Ethics (COMPLETE)
**Files**: `safety_monitor.py`, `emergency_page.py`

#### Crisis Detection System
- **Keywords Monitored**:
  - Suicide: "kill myself", "end my life", "want to die", "suicide"
  - Self-harm: "hurt myself", "cut myself", "harm myself"
  - Violence: "hurt others", "kill someone", "harm others"
  - Abuse: "being abused", "someone hurts me", "unsafe at home"
- **Risk Levels**: none, low, medium, high, crisis (automatic escalation)
- **Medical Disclaimer**: Triggered for health/diagnosis requests
- **Safety Logging**: All safety events recorded to MongoDB

#### Emergency Resources Page
- **24/7 Crisis Hotlines**:
  - 988 Suicide & Crisis Lifeline (call/text)
  - Crisis Text Line (HOME to 741741)
  - Veterans Crisis Line (988 then Press 1)
  - National Domestic Violence Hotline (1-800-799-7233)
  - RAINN Sexual Assault Hotline (1-800-656-4673)
  - Trevor Project LGBTQ+ Support (1-866-488-7386)
  - SAMHSA Mental Health Helpline (1-800-662-4357)
- **International Resources**: findahelpline.com, IASP directory
- **UI**: Clean Streamlit native components with color-coded sections

#### Safety Features Integration
- Automatic crisis response when keywords detected
- Medical disclaimer on health advice requests
- Visible emergency button in app header
- User consent disclaimer before first chat
- Safety warnings on login page
- Crisis response overrides normal chatbot output

### ✅ Phase 7: User Study & Validation (COMPLETE)
**Files**: `feedback_system.py`, `analytics_dashboard.py`, `performance_monitor.py`, `satisfaction_survey.py`

#### Feedback & Analytics System
- **Real-time Feedback**: Thumbs up/down/neutral buttons on every response
- **Analytics Dashboard**: Interactive Plotly visualizations
  - Feedback distribution pie charts
  - Survey rating bar charts
  - Emotion-based feedback analysis
  - Recent feedback viewer with details
- **Data Export**: CSV export for external analysis
- **User History**: Personal feedback tracking per user

#### Performance Monitoring
- **System Metrics**: CPU, memory, disk usage with psutil
- **Response Time Tracking**:
  - Emotion detection time (DistilBERT)
  - LLM response time (LLAMA 3.3)
  - End-to-end response time
- **MongoDB Logging**: All performance metrics stored
- **Performance Summary**: Average times, error rates, uptime

#### Satisfaction Surveys
- **5-Point Rating Scales**:
  - Overall satisfaction
  - Empathy and compassion
  - Helpfulness
  - Ease of use
- **Recommendation Question**: Would recommend (Yes/No/Maybe)
- **Qualitative Feedback**: Comments and suggestions
- **Inline Prompts**: Survey triggers every 15 messages

### ✅ Phase 8: Multi-Language Support (COMPLETE)
**Files**: `language_manager.py`

#### Translation System
- **Supported Languages**: 10 languages
  - 🇬🇧 English | 🇪🇸 Spanish | 🇫🇷 French | 🇮🇳 Hindi | 🇨🇳 Chinese
  - 🇸🇦 Arabic | 🇩🇪 German | 🇵🇹 Portuguese | 🇷🇺 Russian | 🇯🇵 Japanese
- **Translation Engine**: Google Translator via deep-translator library
- **Bidirectional Translation**:
  - User input → English (for emotion detection)
  - Bot response → User's language
- **Language Persistence**: User preference saved in MongoDB
- **UI Localization**: Full interface translation (buttons, labels, messages)
- **Seamless Experience**: Transparent translation in chat pipeline

#### Translation Flow
1. User types message in their language
2. System translates to English
3. Emotion detection on English text
4. LLAMA generates English response
5. System translates back to user's language
6. User sees response in their language

### ✅ Phase 9: Advanced Error Handling (COMPLETE)
**Files**: `error_handler.py`

#### Error Management System
- **Retry Logic**: Exponential backoff with 3 retry attempts
  - Initial delay: 1 second
  - Backoff factor: 2x (1s → 2s → 4s)
- **Offline Detection**: Internet connectivity check (5-second timeout)
- **Fallback Responses**: Emotion-specific pre-configured responses
  - 8 emotion types: joy, sadness, anger, fear, anxiety, neutral, love, surprise
- **Error Logging**: 100-entry circular buffer with timestamps
- **User-Friendly Messages**: Converts technical errors to clear notifications

#### Supported Error Types
1. **Timeout Errors**: Request timeout handling
2. **Rate Limit**: API rate limit exceeded
3. **Network Errors**: Connection issues
4. **API Errors**: General API failures
5. **Database Errors**: MongoDB connection failures
6. **Translation Errors**: Translation service failures
7. **LLM Errors**: Response generation failures
8. **General Errors**: Catch-all for unexpected issues

#### UI Error Notifications
- 🔌 **Offline Mode**: No internet connection
- ⏳ **Rate Limit**: Too many requests warning
- 🌐 **Connection Issue**: Network problems
- ℹ️ **Fallback Response**: Using pre-configured response

#### Error Handling Features
- Automatic retry on transient failures
- Graceful degradation (never crashes)
- Offline mode with coping strategies
- Error tracking and analytics
- Crisis fallback with emergency resources

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+ (tested on Python 3.13)
- MongoDB Atlas account (or local MongoDB)
- Groq API key (free at [console.groq.com](https://console.groq.com))
- 8GB+ RAM
- CUDA-capable GPU (optional but recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/Ayush8905/Emotion-Aware-Conversational-AI-for-mental-Health-Support.git
cd bot\ 2
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch>=2.0.0` - PyTorch for deep learning
- `transformers>=4.30.0` - Hugging Face transformers
- `streamlit==1.41.1` - Web interface
- `pymongo==4.16.0` - MongoDB driver
- `groq>=0.9.0` - LLAMA 3.3 API client
- `bcrypt==5.0.0` - Password hashing
- `plotly>=5.17.0` - Interactive data visualizations
- `psutil>=5.9.0` - System performance monitoring
- `deep-translator>=1.11.0` - Multi-language translation
- `requests>=2.31.0` - HTTP requests for error handling
- `pandas`, `numpy`, `scikit-learn` - Data processing

### Step 4: Configure Environment

Create a `.env` file in the project root (or set environment variables):

```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/
MONGODB_DATABASE=mental_health_chatbot

# Groq API Configuration
GROQ_API_KEY=gsk_your_api_key_here

# Optional: Model Configuration
MODEL_PATH=models/best_model
```

**Get Free API Keys:**
1. **Groq API**: Sign up at [console.groq.com](https://console.groq.com) (free tier available)
2. **MongoDB Atlas**: Create free cluster at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)

### Step 5: Launch the Application

```bash
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**

### Step 6: Create Account & Start Using All Features

1. **Login/Signup**
   - Click "Create New Account" on login page
   - Enter username and password
   - Accept safety disclaimer

2. **Select Your Language** 🌍 (New!)
   - Choose from 10 languages in the sidebar
   - Language preference is saved automatically
   - All messages and UI translated

3. **Chat with AI Therapist**
   - Start conversations about your feelings and emotions
   - Receive empathetic, context-aware responses in your language
   - AI detects emotions in real-time (28 categories)

4. **Provide Feedback**
   - Rate every bot response with 👍 Positive, 👎 Negative, or 😐 Neutral
   - Your feedback helps improve the system

5. **View Analytics Dashboard**
   - Click the 📊 Analytics button in the header
   - View feedback distribution charts
   - See survey rating statistics
   - Analyze emotion-based feedback patterns
   - Export data to CSV for further analysis

6. **Complete Satisfaction Surveys**
   - Surveys appear automatically every 15 messages
   - Rate overall satisfaction, empathy, helpfulness, and ease of use
   - Provide comments and suggestions

7. **Access Emergency Resources**
   - Click 🆘 Emergency button for crisis hotlines
   - 24/7 support lines always available (translated to your language)

8. **View Conversation History**
   - Access past conversations
   - Review emotional trends
   - Export chat history
   - Continue previous conversations

---

## 💡 Using the Application Features

### 1. **Chat Interface**
- Type your message in the chat input (any language!)
- AI detects your emotion (28 categories)
- Receive empathetic, personalized responses in your language
- All conversations saved to your account
- Auto-clearing text input with Enter key support

### 2. **Multi-Language Support** 🌍 (Phase 8)
- Click language dropdown in sidebar
- Select from 10 languages
- System translates your messages and bot responses
- Language preference persists across sessions
- Full UI translation (buttons, labels, messages)

**Translation Flow:**
```
User Input (Spanish) → English → Emotion Detection → 
Response Generation → Spanish → User sees Spanish response
```

### 3. **Error Handling** ⚡ (Phase 9)
- Automatic retry on failures (3 attempts with backoff)
- Offline mode detection with helpful messages
- Fallback responses when AI is unavailable
- Clear error notifications (not technical jargon)
- System never crashes - graceful degradation

**Error Indicators:**
- 🔌 Offline Mode: No internet connection
- ⏳ Rate Limit: Too many requests
- 🌐 Connection Issue: Network problems
- ℹ️ Fallback Response: Using pre-configured response

### 4. **Feedback System**
- After each bot response, you'll see feedback buttons: 👍 👎 😐
- Click to rate the quality and helpfulness of responses
- Feedback is saved to MongoDB for system improvement

### 5. **Analytics Dashboard**
- Click the **📊 Analytics** button in the header
- View interactive Plotly charts showing:
  - Overall feedback distribution (pie chart)
  - Survey rating trends (bar chart)
  - Emotion-based satisfaction rates (stacked bar)
  - Recent feedback history with details
- Export data to CSV for external analysis
- View personal feedback statistics

### 6. **Satisfaction Surveys**
- Complete surveys prompted every 15 messages
- Rate 5 aspects: satisfaction, empathy, helpfulness, ease of use
- Would you recommend? (Yes/No/Maybe)
- Optional text feedback
- Results visible in analytics dashboard

### 7. **Emergency Resources**
- Click 🆘 Emergency button anytime
- Access 24/7 crisis hotlines
- International resources available
- Crisis detection triggers automatic resource display

### 8. **Conversation History**
- View all past conversations with timestamps
- Continue previous conversations
- Delete individual or all conversations
- Export conversations to JSON files
- Review emotional patterns over time
- Rate on 5-point scales:
  - Overall satisfaction
  - Empathy and compassion
  - Helpfulness of responses
  - Ease of use
- Answer "Would you recommend?" (Yes/No/Maybe)
- Provide qualitative feedback and suggestions

### 5. **Performance Monitoring** ⭐ New!
- Real-time system metrics (CPU, memory, disk)
- Response time tracking:
  - Emotion detection time
  - LLM response time
  - End-to-end response time
- All metrics logged to MongoDB
- Performance summaries available in analytics

### 6. **Emergency Resources**
- Click 🆘 Emergency button for immediate crisis support
- Access 24/7 hotlines for suicide prevention, domestic violence, etc.
- International resources available

---

## 🧪 Testing Individual Components

### Test Emotion Detection Only
```bash
python inference.py
```
Interactive mode for testing emotion classification on sample texts.

### Test Chatbot Pipeline (Terminal)
```bash
python test_chatbot.py
```
Command-line testing of end-to-end chatbot with emotion + response generation.

### Test MongoDB Connection
```bash
python test_mongodb_connection.py
```
Verify database connectivity and authentication.

### Run Diagnostics
```bash
python mongodb_diagnostics.py
```
Complete system health check.
python inference.py
```

**Example Output:**
```
=== Emotion Detection System - Sample Predictions ===

Text: "I am so happy today!"
Top predictions:
  1. joy (94.23%)
  2. optimism (3.45%)
  3. excitement (1.12%)

================================================================================
Interactive Mode Started! Type your text to analyze emotions.
Type 'quit' or 'exit' to stop.
================================================================================

Enter text: I love this project!
Detected Emotion: love (87.65%)
```

---

## 📖 Full Pipeline Usage Guide

**Note:** The pre-trained model is already available. You only need to run the full pipeline if you want to retrain or customize the model.

### **STEP 1: Data Preprocessing** (Optional - for retraining)

```bash
python data_preprocessing.py
```

**What it does**:
- Loads `go_emotions_dataset (1).csv`
- Analyzes dataset statistics
- Cleans and preprocesses text
- Creates train/val/test splits
- Saves processed data to `processed_data/`

**Output**:
```
processed_data/
├── train.csv
├── val.csv
├── test.csv
└── label_mapping.json
```

---

### **STEP 2: Model Training** (Optional - for retraining)

```bash
python train_emotion_classifier.py
```

**What it does**:
- Loads preprocessed data
- Initializes DistilBERT model
- Fine-tunes on emotion classification
- Monitors validation performance
- Saves best model checkpoint

**Training Configuration**:
- Model: DistilBERT-base-uncased
- Batch Size: 32
- Max Length: 128 tokens
- Epochs: 3
- Learning Rate: 2e-5
- Optimizer: AdamW

**Output**:
```
models/
├── best_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── vocab.txt
├── best_model_metrics.json
└── test_results.json
```

**Expected Training Time**:
- CPU: ~4-6 hours
- GPU (CUDA): ~30-45 minutes

---

### **STEP 3: Model Evaluation** (Optional)

```bash
python evaluate_model.py
```

**What it does**:
- Loads trained model
- Evaluates on test set
- Generates performance metrics
- Creates visualization plots
- Performs error analysis

**Output**:
```
models/
├── confusion_matrix.png (if implemented)
├── classification_report.txt
└── test_results.json
```

---

### **STEP 4: Interactive Testing** (Main Usage)

```bash
python inference.py
```

**What it does**:
- Loads the pre-trained model from `models/best_model/`
- Runs sample predictions on predefined texts
- Starts interactive terminal mode
- Allows real-time emotion detection

**Sample Predictions Included:**
1. "I am so happy today!"
2. "This makes me really angry"
3. "I'm feeling a bit nervous about the presentation"
4. "That's absolutely hilarious!"
5. "I'm disappointed with the results"

**Interactive Mode:**
- Type any text to analyze emotions
- Get top emotion predictions with confidence scores
- Type 'quit' or 'exit' to stop

---

## 📈 Model Architecture Details

### Why DistilBERT?

**Advantages**:
1. **Efficiency**: 40% smaller than BERT-base
2. **Speed**: 60% faster inference
3. **Performance**: Retains 97% of BERT's capabilities
4. **Memory**: Lower GPU memory requirements

### Model Components

```
Input Text
    ↓
Tokenization (WordPiece)
    ↓
DistilBERT Encoder (6 layers)
    - Self-attention mechanisms
    - Feed-forward networks
    - Layer normalization
    ↓
[CLS] Token Representation
    ↓
Classification Head (Linear + Softmax)
    ↓
Emotion Probabilities (28 classes)
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max Length | 128 | Covers 95% of samples |
| Batch Size | 32 | Balance speed/memory |
| Learning Rate | 2e-5 | Standard for fine-tuning |
| Warmup Steps | 10% | Stable training start |
| Weight Decay | 0.01 | Regularization |
| Epochs | 3 | Prevent overfitting |

---

## 📊 Expected Performance

### Baseline Metrics

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 65-75% |
| Weighted F1 | 63-73% |
| Precision | 64-74% |
| Recall | 65-75% |

**Note**: Performance varies by emotion category. Common emotions (joy, sadness, anger) typically achieve higher F1 scores (75-85%) than rare emotions (grief, remorse).

---

## 🔧 Troubleshooting

### Issue: Virtual Environment Activation Failed (Windows PowerShell)

**Error**: "Execution of scripts is disabled on this system"

**Solution**: 
```bash
# Option 1: Use Command Prompt instead
.venv\Scripts\activate.bat

# Option 2: Set PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Option 3: Run directly without activation
.\.venv\Scripts\python.exe inference.py
```

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or use CPU
```python
# In train_emotion_classifier.py
CONFIG = {
    'batch_size': 16,  # Reduce from 32
    ...
}

# Or force CPU usage
device = torch.device('cpu')
```

### Issue: Slow Training on CPU

**Solutions**:
1. Use the pre-trained model (recommended - no training needed!)
2. Use cloud GPU (Google Colab, Kaggle)
3. Reduce dataset size for prototyping

### Issue: Model Not Found

**Error**: "Model directory not found"

**Solution**: 
- Ensure `models/best_model/` exists with all files
- Check that you're in the correct directory
- If model is missing, run the full pipeline:
  ```bash
  python data_preprocessing.py
  python train_emotion_classifier.py
  ```

### Issue: Import Errors

**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Path Errors on Windows

**Solution**: Use forward slashes or raw strings
```python
# Use forward slashes
model_path = "models/best_model"

# Or raw strings
model_path = r"models\best_model"
```

---

## 📁 Complete Project Structure

```
bot 2/
├── 🌐 Web Application
│   ├── app.py                          # Main Streamlit web interface (6 pages)
│   ├── emergency_page.py               # Crisis resources & hotlines page
│   ├── analytics_dashboard.py          # 📊 Interactive analytics & visualizations (Phase 7)
│   ├── satisfaction_survey.py          # Survey system for user feedback (Phase 7)
│   └── app_backup.py                   # Backup of previous app version
│
├── 🧠 AI Models & Pipeline
│   ├── chatbot_pipeline.py             # End-to-end chatbot orchestration
│   ├── emotion_detector.py             # Emotion detection wrapper
│   ├── response_generator.py           # LLAMA 3.3 response generation
│   ├── inference.py                    # Standalone emotion testing
│   └── safety_monitor.py               # Crisis detection & safety system
│
├── 🗄️ Database & Storage
│   ├── database_manager.py             # MongoDB operations (users, auth)
│   ├── conversation_storage.py         # Chat history management
│   ├── feedback_system.py              # 👍👎😐 Feedback collection backend (Phase 7)
│   └── conversations/                  # JSON exports of chat sessions
│       └── conversation_20260109_142612.json
│
├── 📊 Monitoring & Analytics (Phase 7)
│   ├── performance_monitor.py          # System metrics & response time tracking
│   ├── feedback_system.py              # Feedback collection & analysis
│   ├── analytics_dashboard.py          # Plotly charts & data visualization
│   └── satisfaction_survey.py          # User satisfaction surveys
│
├── 🎓 Training & Evaluation
│   ├── data_preprocessing.py           # GoEmotions dataset preparation
│   ├── train_emotion_classifier.py     # DistilBERT fine-tuning
│   ├── evaluate_model.py               # Model performance analysis
│   └── run_pipeline.py                 # Complete training pipeline runner
│
├── 🧪 Testing & Diagnostics
│   ├── test_chatbot.py                 # Terminal-based chatbot testing
│   ├── test_enhanced_detection.py      # Emotion detection tests
│   ├── test_mongodb_connection.py      # Database connectivity check
│   ├── test_response_setup.py          # Response generator validation
│   └── mongodb_diagnostics.py          # System health diagnostics
│
├── 📦 Data & Models
│   ├── go_emotions_dataset (1).csv     # Original GoEmotions dataset
│   ├── processed_data/                 # Preprocessed train/val/test splits
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── label_mapping.json
│   └── models/                         # Pre-trained emotion classifier
│       ├── best_model/                 # DistilBERT fine-tuned weights
│       │   ├── config.json
│       │   ├── model.safetensors
│       │   ├── tokenizer.json
│       │   ├── tokenizer_config.json
│       │   ├── special_tokens_map.json
│       │   └── vocab.txt
│       ├── best_model_metrics.json     # Training performance
│       ├── classification_report.txt   # Detailed evaluation
│       └── test_results.json           # Test set results
│
├── 📋 Documentation & Config
│   ├── README.md                       # This comprehensive guide
│   ├── requirements.txt                # Python dependencies
│   └── .env                           # Environment variables (create this)
│
└── 🗂️ Cache
    └── __pycache__/                    # Python bytecode cache
```

### Key Files Explained

| File | Purpose | Status |
|------|---------|--------|
| `app.py` | Main web application with 6 pages (login, chat, history, emergency, analytics, survey) | ✅ Production Ready |
| `analytics_dashboard.py` | Interactive Plotly charts for feedback & performance visualization | ✅ Phase 7 Complete |
| `feedback_system.py` | MongoDB backend for collecting user ratings & survey data | ✅ Phase 7 Complete |
| `performance_monitor.py` | System metrics tracking (CPU, memory, response times) | ✅ Phase 7 Complete |
| `satisfaction_survey.py` | User satisfaction surveys with 5-point rating scales | ✅ Phase 7 Complete |
| `chatbot_pipeline.py` | Orchestrates emotion detection + response generation | ✅ Complete |
| `safety_monitor.py` | Crisis keyword detection & emergency response | ✅ Complete |
| `emergency_page.py` | 24/7 hotlines & crisis resources display | ✅ Complete |
| `database_manager.py` | User authentication & MongoDB operations | ✅ Complete |
| `conversation_storage.py` | Chat history persistence & retrieval | ✅ Complete |
| `response_generator.py` | LLAMA 3.3 API integration for empathetic responses | ✅ Complete |
| `emotion_detector.py` | DistilBERT emotion classification wrapper | ✅ Complete |
| `train_emotion_classifier.py` | Model training (optional - pre-trained included) | ✅ Complete |

---

## 🎯 Feature Highlights

### 1️⃣ Emotion Detection System
- **28 Emotion Categories**: joy, sadness, anger, fear, love, gratitude, optimism, nervousness, pride, confusion, and 18 more
- **Model**: Fine-tuned DistilBERT (97% of BERT performance, 60% faster)
- **Accuracy**: 65-75% on GoEmotions test set
- **Real-time**: <100ms inference on GPU

### 2️⃣ Empathetic Response Generation
- **LLM**: LLAMA 3.3 70B via Groq (ultra-fast inference)
- **System Prompt**: Mental health support specialist role
- **Context-Aware**: Uses detected emotions to tailor responses
- **Tone**: Warm, non-judgmental, validating, professional
- **Techniques**: Active listening, validation, coping strategies

### 3️⃣ Safety & Crisis Management
- **Automatic Detection**: Monitors for suicide, self-harm, violence, abuse keywords
- **Risk Levels**: 5-tier system (none → low → medium → high → crisis)
- **Emergency Override**: Crisis response replaces normal chatbot output
- **24/7 Hotlines**: 988 Lifeline, Crisis Text Line, Veterans Crisis Line
- **Medical Disclaimers**: Activated for diagnosis/medication requests
- **Event Logging**: All safety incidents recorded to database

### 4️⃣ User Experience
- **Session Persistence**: Chat history survives page refreshes
- **Auto-clear Input**: Text field clears automatically after send
- **Conversation History**: Browse and continue previous chats
- **New Chat Button**: Start fresh conversations instantly
- **Emergency Access**: One-click access to crisis resources
- **Mobile Responsive**: Works on all screen sizes

### 5️⃣ Security & Privacy
- **Password Hashing**: bcrypt with 12 salt rounds
- **Session Tokens**: Secure session management
- **Data Encryption**: MongoDB Atlas encryption at rest
- **No PII Storage**: Minimal personally identifiable information
- **GDPR Considerations**: Conversation export & deletion support

---

## 🔬 Technical Deep Dive

### Emotion Detection Pipeline

```
User Input Text
    ↓
Text Preprocessing (lowercase, URL removal)
    ↓
Tokenization (DistilBERT WordPiece)
    ↓
DistilBERT Encoder (6 transformer layers)
    - Self-attention mechanisms
    - Feed-forward networks
    - Layer normalization
    ↓
[CLS] Token Representation (768-dim)
    ↓
Classification Head (Linear + Softmax)
    ↓
Emotion Probabilities (28 classes)
    ↓
Top Emotion Selected
```

### Response Generation Pipeline

```
Detected Emotion + User Message
    ↓
System Prompt Construction
    - Role: Mental health support specialist
    - Tone: Empathetic, non-judgmental
    - Emotion Context: Inject detected emotion
    ↓
LLAMA 3.3 70B API Call (Groq)
    - Temperature: 0.7
    - Max Tokens: 500
    - Stream: False
    ↓
Empathetic Response
    - Validation of feelings
    - Active listening
    - Coping suggestions
    ↓
Safety Check
    - Medical disclaimer if needed
    - Crisis response if triggered
    ↓
Display to User
```

### Safety Monitoring Flow

```
User Message Received
    ↓
Keyword Scanning
    - Suicide keywords (10+)
    - Self-harm patterns
    - Violence indicators
    - Abuse mentions
    ↓
Risk Assessment
    - Crisis: Immediate danger keywords
    - High: Multiple warning signs
    - Medium: Single warning keyword
    - Low: Indirect distress
    - None: No safety concerns
    ↓
Response Decision
    - Crisis → Override with emergency resources
    - High/Medium → Add safety disclaimer to response
    - Low → Mention available resources
    - None → Normal chatbot response
    ↓
Event Logging
    - Timestamp
    - Username
    - Detected concerns
    - Risk level
    - Save to MongoDB safety_events collection
```

### Model Training Details

**Hyperparameters:**
```python
CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'num_labels': 28,
    'max_length': 128,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'epochs': 3,
    'warmup_steps': 0.1,
    'weight_decay': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

**Training Process:**
1. Load preprocessed GoEmotions dataset (148,219 training samples)
2. Initialize DistilBERT with classification head
3. Freeze first 3 layers (optional - currently trains all layers)
4. Train with cross-entropy loss
5. Validate every epoch
6. Save best model based on validation F1-score
7. Final test set evaluation

**Performance Metrics:**
- **Accuracy**: 68-73% (expected)
- **Weighted F1**: 66-71%
- **Training Time**: 30-45 minutes on GPU, 4-6 hours on CPU
- **Model Size**: ~260MB (DistilBERT weights + classification head)

### Database Schema

**MongoDB Collections:**

```javascript
// users collection
{
  "_id": ObjectId("..."),
  "username": "john_doe",
  "password_hash": "$2b$12$...",  // bcrypt hash
  "created_at": ISODate("2026-01-12T10:30:00Z"),
  "last_login": ISODate("2026-01-12T15:45:00Z")
}

// conversations collection
{
  "_id": ObjectId("..."),
  "username": "john_doe",
  "conversation_id": "conv_1736691234",
  "messages": [
    {
      "role": "user",
      "content": "I'm feeling anxious about work",
      "timestamp": ISODate("2026-01-12T15:45:10Z"),
      "emotion": "nervousness",
      "confidence": 0.87
    },
    {
      "role": "assistant",
      "content": "I hear that you're feeling anxious...",
      "timestamp": ISODate("2026-01-12T15:45:15Z")
    }
  ],
  "started_at": ISODate("2026-01-12T15:45:10Z"),
  "updated_at": ISODate("2026-01-12T15:50:30Z"),
  "active": true
}

// safety_events collection
{
  "_id": ObjectId("..."),
  "username": "john_doe",
  "timestamp": ISODate("2026-01-12T16:00:00Z"),
  "message": "I've been thinking about hurting myself",
  "concerns": ["self-harm"],
  "risk_level": "high",
  "response_sent": true
}
```

---

## ⚠️ Known Limitations & Ethical Considerations

### Current Limitations

1. **Not a Replacement for Professional Care**
   - This is a research prototype for educational purposes
   - Should not be used as primary mental health treatment
   - Cannot provide diagnosis or prescribe medication
   - No substitute for licensed therapists or psychiatrists

2. **Emotion Detection Accuracy**
   - 68-73% accuracy means 27-32% misclassifications
   - Struggles with sarcasm, idioms, and cultural context
   - Multi-emotion texts may be oversimplified
   - Biased toward emotions common in training data

3. **Response Generation**
   - LLM may occasionally generate inappropriate content
   - No guarantee of consistency across conversations
   - May reinforce biases present in training data
   - Limited memory beyond current conversation

4. **Safety System**
   - Keyword-based detection has false positives/negatives
   - May miss subtle crisis indicators
   - Cannot physically intervene in emergencies
   - Relies on user accessing emergency resources

### Ethical Guidelines

**DO:**
✅ Include clear disclaimers about system limitations  
✅ Provide emergency hotlines prominently  
✅ Store data securely with encryption  
✅ Obtain informed consent before data collection  
✅ Allow users to delete their data  
✅ Log safety events for system improvement  
✅ Regularly audit for bias and fairness  

**DON'T:**
❌ Claim medical accuracy or clinical validity  
❌ Store sensitive health information without proper safeguards  
❌ Deploy without crisis escalation mechanisms  
❌ Use in high-stakes decision-making (e.g., involuntary commitment)  
❌ Collect more data than necessary  
❌ Share user data with third parties  
❌ Ignore accessibility requirements  

### Compliance Considerations

**HIPAA (USA)**: This system is **NOT HIPAA-compliant** as-is. For healthcare deployment:
- Implement Business Associate Agreements (BAA)
- Add audit logging for all data access
- Encrypt data in transit and at rest (already done via MongoDB Atlas)
- Implement role-based access controls
- Regular security audits

**GDPR (EU)**: Partially compliant. For full compliance:
- Add explicit consent mechanisms
- Implement "right to be forgotten" (data deletion)
- Provide data portability (export feature exists)
- Document data processing activities
- Appoint Data Protection Officer (DPO)

---

## 🧪 Testing & Quality Assurance

### Automated Testing

**Run All Tests:**
```bash
# Test emotion detection
python test_enhanced_detection.py

# Test chatbot pipeline
python test_chatbot.py

# Test database connectivity
python test_mongodb_connection.py

# Test response generation
python test_response_setup.py

# System diagnostics
python mongodb_diagnostics.py
```

### Manual Testing Checklist

**Emotion Detection:**
- [ ] Detects positive emotions (joy, love, gratitude)
- [ ] Detects negative emotions (sadness, anger, fear)
- [ ] Handles neutral text appropriately
- [ ] Processes long inputs (>100 words)
- [ ] Works with misspellings and typos

**Response Generation:**
- [ ] Generates empathetic responses
- [ ] Maintains conversation context
- [ ] Avoids harmful or offensive content
- [ ] Provides actionable coping strategies
- [ ] Respects user boundaries

**Safety Features:**
- [ ] Detects suicide keywords and shows crisis response
- [ ] Detects self-harm mentions
- [ ] Shows medical disclaimers for health queries
- [ ] Emergency button accessible from all pages
- [ ] Crisis resources display correctly

**User Interface:**
- [ ] Login/signup works correctly
- [ ] Password hashing prevents plaintext storage
- [ ] Session persists after refresh
- [ ] Text input clears after send
- [ ] New Chat button creates fresh session
- [ ] Conversation history loads correctly
- [ ] Mobile responsive on various screen sizes

**Database Operations:**
- [ ] Users stored correctly in MongoDB
- [ ] Conversations saved with all metadata
- [ ] Password verification works
- [ ] Duplicate username prevention
- [ ] Connection error handling

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Set environment variables in Streamlit Cloud dashboard
4. Deploy with one click

**Pros**: Free, automatic HTTPS, easy updates  
**Cons**: Limited resources, public apps only

### Option 2: Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t mental-health-chatbot .
docker run -p 8501:8501 --env-file .env mental-health-chatbot
```

### Option 3: Cloud Platforms

**AWS EC2:**
```bash
# On Ubuntu instance
sudo apt update && sudo apt install python3-pip
pip3 install -r requirements.txt
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

**Google Cloud Run**: Supports Docker containers with auto-scaling  
**Azure Web Apps**: Python app hosting with GitHub integration  
**Heroku**: Easy deployment with Procfile

### Production Checklist

- [ ] Set `DEBUG=False` in environment
- [ ] Use production MongoDB cluster (not free tier)
- [ ] Implement rate limiting on API calls
- [ ] Add CDN for static assets
- [ ] Set up monitoring (Sentry, New Relic)
- [ ] Configure logging (CloudWatch, Stackdriver)
- [ ] Implement HTTPS/SSL
- [ ] Add health check endpoint
- [ ] Set up automated backups
- [ ] Create disaster recovery plan

---

## 📚 References

1. **GoEmotions Dataset**: Demszky et al. (2020)
2. **DistilBERT**: Sanh et al. (2019)
3. **BERT**: Devlin et al. (2018)
4. **Mental Health NLP**: Calvo et al. (2017)

---

## ⚠️ Ethical Considerations

**Important Disclaimers**:
1. This is a **research prototype**, not a clinical tool
2. **Not a replacement** for professional mental health care
3. Should include **crisis resource information**
4. Requires **informed consent** for user studies
5. Must comply with **HIPAA/GDPR** for real deployments

---

## 🤝 Contributing & Future Development

### Potential Improvements

**Short-term (1-3 months)**:
- [ ] Add conversation export as PDF/JSON
- [ ] Implement "Edit message" functionality
- [ ] Add voice input/output (text-to-speech)
- [ ] Multi-language support (Spanish, French, Chinese)
- [ ] Dark mode UI option
- [ ] Emoji reaction buttons for responses
- [ ] Chat search functionality

**Medium-term (3-6 months)**:
- [ ] Multi-turn conversation memory with vector database
- [ ] Fine-tune LLAMA on mental health conversations
- [ ] A/B testing framework for response quality
- [ ] User feedback collection (thumbs up/down)
- [ ] Mood tracking over time with visualization
- [ ] Integration with wearables (heart rate, sleep data)
- [ ] Therapist dashboard for monitoring (with consent)

**Long-term (6-12 months)**:
- [ ] Clinical trials with licensed therapists
- [ ] HIPAA-compliant version for healthcare
- [ ] Real-time video session support
- [ ] Automated crisis escalation to human counselors
- [ ] Personalized coping strategy recommendations
- [ ] Group therapy chatroom features
- [ ] Mobile app (React Native, Flutter)

### How to Contribute

This is an educational research project. If you'd like to improve it:

1. **Report Issues**: Open GitHub issues for bugs or feature requests
2. **Submit Pull Requests**: Fork, create feature branch, submit PR
3. **Test New Models**: Try BERT-large, RoBERTa, GPT-4, Claude
4. **Add Datasets**: Integrate other emotion datasets (EmoInt, ISEAR)
5. **Improve Safety**: Enhance crisis detection algorithms
6. **UI/UX**: Design better interfaces, improve accessibility

---

## 📚 References & Resources

### Academic Papers

1. **GoEmotions Dataset**  
   Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*  
   [arXiv:2005.00547](https://arxiv.org/abs/2005.00547)

2. **DistilBERT**  
   Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT*  
   [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)

3. **BERT**  
   Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers*  
   [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

4. **Mental Health NLP**  
   Calvo, R. A., et al. (2017). *Natural language processing in mental health applications*

### Tools & Frameworks

- **Hugging Face Transformers**: [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **MongoDB Atlas**: [mongodb.com/docs/atlas](https://www.mongodb.com/docs/atlas/)
- **Groq API**: [console.groq.com/docs](https://console.groq.com/docs)
- **PyTorch**: [pytorch.org/docs](https://pytorch.org/docs)

### Mental Health Resources

- **988 Suicide & Crisis Lifeline**: [988lifeline.org](https://988lifeline.org)
- **Crisis Text Line**: [crisistextline.org](https://www.crisistextline.org)
- **SAMHSA National Helpline**: [samhsa.gov/find-help/national-helpline](https://www.samhsa.gov/find-help/national-helpline)

---

## 📧 Support

**GitHub Repository**: [Ayush8905/Emotion-Aware-Conversational-AI-for-mental-Health-Support](https://github.com/Ayush8905/Emotion-Aware-Conversational-AI-for-mental-Health-Support)

For questions, issues, or collaboration:
1. Open a GitHub issue
2. Check existing documentation
3. Review troubleshooting guide
4. Test with diagnostic scripts

---

## ✅ Project Completion Status

### Phase 1-3: Emotion Detection ✅ COMPLETE
- [x] GoEmotions dataset analysis and preprocessing
- [x] DistilBERT fine-tuning on 28 emotions
- [x] Model evaluation and metrics
- [x] Real-time inference system
- [x] Pre-trained model included

### Phase 4: Response Generation ✅ COMPLETE
- [x] LLAMA 3.3 API integration via Groq
- [x] Empathetic system prompt design
- [x] Context-aware response generation
- [x] Emotion-conditioned responses

### Phase 5A: Database Backend ✅ COMPLETE
- [x] MongoDB Atlas integration
- [x] User authentication with bcrypt
- [x] Conversation storage and retrieval
- [x] Session management

### Phase 5B: Web Interface ✅ COMPLETE
- [x] Streamlit UI implementation
- [x] Login/signup system
- [x] Chat interface with history
- [x] Session persistence across refreshes
- [x] Mobile-responsive design

### Phase 6: Safety & Ethics ✅ COMPLETE
- [x] Crisis keyword detection system
- [x] Risk level assessment
- [x] Emergency resources page
- [x] Medical disclaimer system
- [x] Safety event logging
- [x] User consent disclaimers

### ✅ Phase 7: User Study & Validation (COMPLETE)
**Files**: `feedback_system.py`, `analytics_dashboard.py`, `performance_monitor.py`, `satisfaction_survey.py`

#### Feedback Collection System
- **Real-time Feedback**: Thumbs up/down/neutral buttons on every bot response
- **MongoDB Storage**: Dedicated collections for feedback, surveys, and performance logs
- **Emotion Tracking**: Feedback categorized by detected emotion types
- **User Association**: All feedback linked to user sessions for analysis
- **Features**:
  - record_response_feedback() - Capture user ratings (positive/negative/neutral)
  - get_feedback_statistics() - Aggregated metrics and satisfaction scores
  - get_emotion_feedback_breakdown() - Feedback analysis by emotion category
  - export_feedback_data() - CSV export for external analysis

#### Analytics Dashboard
- **Visualizations**: Interactive Plotly charts for feedback analysis
  - Pie charts for feedback distribution
  - Bar charts for survey rating trends
  - Stacked bar charts for emotion-based feedback
  - Time-series analysis of user satisfaction
- **Metrics Displayed**:
  - Total responses and feedback counts
  - Satisfaction percentages by category
  - Emotion-specific approval ratings
  - Recent feedback with expandable details
- **User Features**:
  - Personal feedback history
  - Export functionality for reports
  - Session-based authentication
  - Real-time data updates

#### Performance Monitoring
- **System Metrics**: CPU, memory, and disk usage tracking with psutil
- **Response Time Tracking**:
  - Emotion detection time (DistilBERT inference)
  - LLM response time (LLAMA 3.3 API)
  - Total end-to-end response time
- **Performance Logging**: All metrics stored in MongoDB for analysis
- **Features**:
  - log_response_time() - Track complete chatbot response cycles
  - log_emotion_detection_time() - Monitor model inference speed
  - log_llm_response_time() - Track API call latency
  - get_system_stats() - Real-time system health metrics
  - get_performance_summary() - Aggregate performance statistics

#### Satisfaction Surveys
- **Survey Types**:
  - Full post-conversation survey with 5-point rating scales
  - Inline periodic prompts (every 15 messages)
  - Optional skip functionality
- **Rating Categories**:
  - Overall satisfaction (1-5 scale)
  - Empathy and compassion (1-5 scale)
  - Helpfulness of responses (1-5 scale)
  - Ease of use (1-5 scale)
  - Would recommend (Yes/No/Maybe)
- **Qualitative Feedback**:
  - Comments text area
  - Suggestions for improvement
  - Open-ended feedback collection

#### Integration Features
- **UI Integration**: Seamlessly embedded in main app.py
  - Analytics button in header (6-column layout)
  - Feedback buttons below each bot response
  - Survey prompts at message milestones
  - Dedicated analytics page with charts
- **Session State Management**: Tracks feedback per message to prevent duplicates
- **Performance Instrumentation**: Integrated into chatbot_pipeline.py
- **Export Tools**: CSV download for feedback and survey data

#### Testing & Validation
- [x] Unit tests for feedback system
- [x] Integration tests for analytics dashboard
- [x] Performance monitoring validation
- [x] Survey system testing
- [x] UI/UX manual testing
- [x] Session state persistence testing
- [x] MongoDB collection verification
- [x] Export functionality validation

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Total Lines of Code** | ~5,000+ |
| **Python Files** | 24 |
| **Dataset Size** | 211,742 samples |
| **Model Parameters** | 66M (DistilBERT) + 70B (LLAMA 3.3) |
| **Emotion Categories** | 28 + neutral |
| **Languages Supported** | 10 (Multi-language) |
| **Training Time (GPU)** | 30-45 minutes |
| **Inference Speed** | <100ms per message |
| **Translation Speed** | ~500ms per message |
| **Database Collections** | 6 (users, conversations, safety_events, feedback, surveys, performance_logs) |
| **Crisis Keywords** | 40+ monitored |
| **Emergency Hotlines** | 7 (US) + international |
| **Dependencies** | 19 major packages |
| **UI Pages** | 6 (login, chat, history, emergency, analytics, survey) |
| **Feedback Types** | 3 (positive, negative, neutral) |
| **Survey Metrics** | 5 rating scales + qualitative feedback |
| **Error Types Handled** | 8 (timeout, rate_limit, network, api, database, translation, llm, general) |
| **Retry Attempts** | 3 with exponential backoff |
| **Fallback Responses** | 8 emotion-specific templates |
| **Development Phases** | 9/9 (100% Complete) |

---

## 📄 License & Important Disclaimer

### Educational Use Only

This project is intended for **educational and research purposes only**. It is:
- ❌ NOT FDA-approved
- ❌ NOT a medical device
- ❌ NOT a substitute for professional therapy
- ❌ NOT for commercial use without proper licensing

### Crisis Support Disclaimer

**IMPORTANT**: This AI chatbot is a research prototype developed for academic purposes. It is not a replacement for professional mental health care, diagnosis, or treatment.

**If you are in crisis or experiencing suicidal thoughts:**
- 🆘 **Call 988** (USA) - Suicide & Crisis Lifeline
- 📱 **Text HOME to 741741** - Crisis Text Line
- 🚨 **Call 911** - For immediate life-threatening emergencies
- 🌍 **International**: [findahelpline.com](https://findahelpline.com)

The AI system:
- May provide inaccurate or inappropriate responses
- Cannot provide clinical diagnoses or prescribe medication
- Has limited understanding of complex mental health conditions
- Should not be relied upon for critical health decisions

**Always consult with qualified mental health professionals for proper care.**

### Data Privacy

- User data stored securely in MongoDB Atlas
- Passwords hashed with bcrypt (cannot be reversed)
- Conversations saved for system improvement only
- No data sold or shared with third parties
- Users can request data deletion

### Open Source License

MIT License - See LICENSE file for details

---

## 🎉 Acknowledgments

**Dataset**: Google Research for GoEmotions dataset  
**Models**: Hugging Face for DistilBERT, Meta AI for LLAMA 3  
**Infrastructure**: MongoDB Atlas, Groq API, Streamlit  
**Inspiration**: Mental health professionals and researchers working to improve access to care  

---

**Project Status**: ✅ **Production Ready** (with disclaimers)  
**Last Updated**: January 13, 2026  
**Version**: 2.1.0 (Multi-Language + Advanced Error Handling)  
**Author**: Ayush  
**Repository**: [github.com/Ayush8905/Emotion-Aware-Conversational-AI-for-mental-Health-Support](https://github.com/Ayush8905/Emotion-Aware-Conversational-AI-for-mental-Health-Support)

---

**Built with ❤️ for advancing mental health AI and NLP research**

*Remember: Reaching out for help is a sign of strength, not weakness.*
