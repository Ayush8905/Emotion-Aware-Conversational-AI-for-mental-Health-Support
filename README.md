# Emotion-Aware Conversational AI for Mental Health Support

## 📋 Project Overview

This project implements a **transformer-based emotion detection system** for mental health applications. The system uses the **GoEmotions dataset** (28 emotion categories) and fine-tunes **DistilBERT** for accurate emotion classification from text.

### 🎯 System Capabilities

✅ **Emotion Detection**: Classifies text into 28+ emotion categories  
✅ **Multi-label Support**: Handles complex emotional states  
✅ **High Accuracy**: Achieves strong F1 scores on validation data  
✅ **Fast Inference**: Optimized for real-time predictions  
✅ **Interactive Testing**: User-friendly command-line interface  
✅ **Pre-trained Model**: Ready to use without retraining

---

## 📊 Dataset Information

**Dataset**: GoEmotions (Google Research)
- **Total Samples**: 211,742 Reddit comments
- **Emotion Labels**: 28 categories + neutral
- **Labels**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

---

## 🏗️ System Architecture

### Phase 1: Data Preprocessing (`data_preprocessing.py`)
- Load and analyze GoEmotions dataset
- Text cleaning (URL removal, normalization)
- Convert multi-label to single-label
- Stratified train/val/test split (70/15/15)

### Phase 2: Model Training (`train_emotion_classifier.py`)
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Architecture**: Transformer encoder + classification head
- **Loss Function**: Cross-entropy loss
- **Optimizer**: AdamW with linear warmup
- **Training**: 3 epochs with validation monitoring

### Phase 3: Evaluation (`evaluate_model.py`)
- Compute accuracy, precision, recall, F1-score
- Generate confusion matrix visualization
- Per-class performance analysis
- Error analysis and misclassification patterns

### Phase 4: Inference (`inference.py`)
- Real-time emotion prediction
- Interactive terminal testing
- Batch prediction mode
- Confidence scores for top-K emotions

---

## 🚀 Quick Start (Using Pre-trained Model)

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM

### Step 1: Create Virtual Environment (Recommended)

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

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Step 4: Run Inference (No Training Required!)

The project includes a **pre-trained model** in the `models/best_model/` directory. You can start testing immediately:

```bash
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

## 📁 Project Structure

```
bot 2/
├── go_emotions_dataset (1).csv      # Original dataset
├── data_preprocessing.py             # Data pipeline script
├── train_emotion_classifier.py       # Training script
├── evaluate_model.py                 # Evaluation script
├── inference.py                      # Testing script (READY TO USE)
├── run_pipeline.py                   # Complete pipeline runner
├── requirements.txt                  # Python dependencies
├── README.md                         # This documentation
├── PROJECT_SUMMARY.md                # Project summary
├── QUICKSTART.md                     # Quick start guide
├── TECHNICAL_DOCS.md                 # Technical documentation
├── processed_data/                   # Preprocessed data (generated)
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── label_mapping.json
└── models/                           # Pre-trained model (INCLUDED)
    ├── best_model/
    │   ├── config.json               # Model configuration
    │   ├── model.safetensors         # Trained weights
    │   ├── tokenizer.json            # Tokenizer
    │   ├── tokenizer_config.json
    │   ├── special_tokens_map.json
    │   └── vocab.txt                 # Vocabulary
    ├── best_model_metrics.json       # Training metrics
    ├── classification_report.txt     # Evaluation report
    └── test_results.json             # Test set results
```

---

## 🎯 Model Files Included

The project comes with a **fully trained model** ready for inference:

✅ **Model Weights**: `models/best_model/model.safetensors`  
✅ **Configuration**: `models/best_model/config.json`  
✅ **Tokenizer**: `models/best_model/tokenizer.json`  
✅ **Vocabulary**: `models/best_model/vocab.txt`  
✅ **Metrics**: `models/best_model_metrics.json`  
✅ **Test Results**: `models/test_results.json`

**No training required** - just install dependencies and run [`inference.py`](inference.py)!

---

## 🎓 Methodology & Academic Context

### Problem Statement

Mental health applications require AI systems that can:
1. Detect emotional states from text
2. Generate appropriate empathetic responses
3. Maintain emotional context across conversations
4. Include safety mechanisms

This project addresses **Phase 1-3** of the problem statement:
- ✅ **Phase 1**: Problem analysis and architecture design
- ✅ **Phase 2**: Dataset analysis and preprocessing
- ✅ **Phase 3**: Emotion detection model training

### Why Transformers?

1. **Contextual Understanding**: Self-attention captures long-range dependencies
2. **Transfer Learning**: Pretrained on massive corpora
3. **State-of-the-Art**: Best performance on NLP benchmarks
4. **Flexibility**: Easy fine-tuning for specific tasks

### Training Strategy

**Fine-tuning Approach**:
- Start with pretrained DistilBERT weights
- Add classification head for 28 emotions
- Train only top layers initially (optional)
- Use emotion-specific loss function

**Evaluation Metrics**:
- **Accuracy**: Overall correctness
- **F1-Score**: Balance precision/recall (important for imbalanced classes)
- **Per-class Metrics**: Identify weak emotion categories
- **Confusion Matrix**: Understand error patterns

---

## 🔬 Future Enhancements (Phase 4-7)

### Phase 4: Response Generation
- Integrate DialoGPT/GPT-2 for empathetic responses
- Condition generation on detected emotions
- Implement emotion-aware prompting

### Phase 5: Conversation Memory
- LSTM/Transformer memory for context tracking
- Emotion history across turns
- Dynamic response adaptation

### Phase 6: Safety & Ethics
- Crisis keyword detection (self-harm, suicide)
- Ethical disclaimers
- Escalation to human professionals
- Bias analysis and mitigation

### Phase 7: User Study
- Human evaluation of empathy
- A/B testing response quality
- Mental health professional validation

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

## 🤝 Contributing

This is an educational research project. For improvements:
1. Test on additional emotion datasets
2. Experiment with larger models (BERT, RoBERTa)
3. Implement multi-label classification
4. Add response generation modules

---

## 📧 Support

For questions or issues:
1. Check error messages carefully
2. Review installation steps
3. Verify dataset integrity
4. Ensure sufficient system resources

---

## ✅ Quick Start Checklist

### For Immediate Use (Pre-trained Model):
- [ ] Install Python 3.8+
- [ ] Create virtual environment: `python -m venv .venv`
- [ ] Activate environment (Windows): `.\.venv\Scripts\Activate.ps1` or `.venv\Scripts\activate.bat`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Run inference: `python inference.py`
- [ ] Test with your own text in interactive mode

### For Full Training Pipeline (Optional):
- [ ] Install Python 3.8+
- [ ] Create and activate virtual environment
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Run preprocessing: `python data_preprocessing.py`
- [ ] Train model: `python train_emotion_classifier.py` (requires 30-360 minutes)
- [ ] Evaluate: `python evaluate_model.py`
- [ ] Test interactively: `python inference.py`

---

**Project Status**: ✅ Fully Functional - Pre-trained Model Included  
**Ready to Use**: Run `python inference.py` immediately after installing dependencies  
**Next Steps**: Experiment with response generation (Phase 4) or customize for your use case

---

*Built for NLP research and mental health AI applications*
