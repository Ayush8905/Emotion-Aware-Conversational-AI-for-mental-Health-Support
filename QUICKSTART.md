# 🚀 QUICK START GUIDE

## ⚡ Get Started in 5 Minutes

### Prerequisites Check
```bash
# Check Python version (need 3.8+)
python --version

# Check if pip is installed
pip --version
```

---

## 📦 Installation (2 minutes)

```bash
# Install all dependencies
pip install -r requirements.txt
```

**Expected output**: Installing packages... (takes 1-2 minutes)

---

## 🎯 Training the Model (30-45 minutes on GPU, 4-6 hours on CPU)

### Option 1: Run Complete Pipeline (Recommended)
```bash
python run_pipeline.py
```

This will automatically:
1. ✅ Preprocess data
2. ✅ Train model  
3. ✅ Evaluate performance
4. ✅ Launch interactive testing

### Option 2: Run Steps Individually

**Step 1: Preprocess Data (~2 minutes)**
```bash
python data_preprocessing.py
```
Output: Creates `processed_data/` folder with train/val/test splits

**Step 2: Train Model (~30-45 min GPU, 4-6 hours CPU)**
```bash
python train_emotion_classifier.py
```
Output: Saves trained model to `models/best_model/`

**Step 3: Evaluate Model (~5 minutes)**
```bash
python evaluate_model.py
```
Output: Generates performance metrics and visualizations

**Step 4: Test Interactively**
```bash
python inference.py
```
Output: Interactive terminal for testing predictions

---

## 💡 Quick Test Examples

Once training is complete, test with these examples:

```
Enter text: I'm so excited about this project!
→ excitement (89%), joy (7%), optimism (3%)

Enter text: This is really frustrating and annoying.
→ annoyance (76%), anger (18%), disappointment (4%)

Enter text: I feel sad and disappointed.
→ sadness (68%), disappointment (24%), grief (6%)

Enter text: Thank you so much for your help!
→ gratitude (91%), admiration (6%), approval (2%)
```

---

## 📊 What to Expect

### Training Progress
```
Epoch 1/3
================================================================================
Train Loss: 1.2456 | Train Acc: 0.6234
Val Loss:   0.9876 | Val Acc:   0.6789
Val F1:     0.6543

✓ New best model saved! (F1: 0.6543)
```

### Final Results
```
TEST SET RESULTS
================================================================================
Accuracy:  0.7012
Precision: 0.6945
Recall:    0.6983
F1 Score:  0.6867
```

---

## 🔍 Verify Everything Works

### 1. Check files exist:
```bash
# After preprocessing
dir processed_data    # Windows
ls processed_data     # Linux/Mac

# After training
dir models\best_model       # Windows
ls models/best_model        # Linux/Mac
```

### 2. Quick sanity test:
```bash
python -c "from transformers import AutoTokenizer; print('✓ Transformers working!')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__} working!')"
```

---

## ⚠️ Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Edit `train_emotion_classifier.py`, line ~220:
```python
CONFIG = {
    'batch_size': 16,  # Change from 32 to 16 or 8
    ...
}
```

### Problem: "ModuleNotFoundError: No module named 'transformers'"
**Solution**: 
```bash
pip install transformers torch pandas scikit-learn
```

### Problem: Training is very slow
**Solutions**:
1. Use GPU (60x faster than CPU)
2. Reduce dataset for testing:
   - Edit `data_preprocessing.py`, add after loading data:
   ```python
   self.df = self.df.sample(n=10000)  # Use only 10k samples
   ```

### Problem: "FileNotFoundError: go_emotions_dataset (1).csv"
**Solution**: Ensure the CSV file is in the same folder as the scripts

---

## 📈 Expected Timeline

| Step | Time (GPU) | Time (CPU) |
|------|-----------|-----------|
| Install dependencies | 2 min | 2 min |
| Data preprocessing | 2 min | 3 min |
| Model training | 30-45 min | 4-6 hours |
| Evaluation | 3 min | 5 min |
| **TOTAL** | **~40 min** | **~5 hours** |

---

## ✅ Success Checklist

After running everything, you should have:

- [x] `processed_data/` folder with train/val/test CSV files
- [x] `models/best_model/` folder with trained model
- [x] `models/confusion_matrix.png` visualization
- [x] `models/per_class_metrics.png` visualization
- [x] `models/evaluation_report.txt` with metrics
- [x] Interactive inference working in terminal

---

## 🎓 Understanding the Output

### During Training:
- **Train Loss**: Should decrease (1.5 → 0.8)
- **Val Accuracy**: Should increase (0.60 → 0.70)
- **Val F1**: Should increase (0.58 → 0.68)

### Final Metrics:
- **Accuracy > 65%**: ✅ Good performance
- **F1 Score > 63%**: ✅ Balanced precision/recall
- **Accuracy < 55%**: ⚠️ May need hyperparameter tuning

---

## 📖 Next Steps

1. **Review Results**: Check `models/evaluation_report.txt`
2. **Analyze Errors**: Read `models/error_analysis.txt`
3. **Test Thoroughly**: Use `inference.py` with diverse examples
4. **Experiment**: Try different models (roberta-base, bert-base)

---

## 💬 Interactive Testing Examples

```bash
$ python inference.py

QUICK TEST - SAMPLE PREDICTIONS
================================================================================
INPUT TEXT: "I am so excited about this!"
DETECTED EMOTIONS:
1. excitement       ████████████████████████████████████████ 91.23%
2. joy              ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  5.67%
3. optimism         █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.45%
```

---

## 🛠️ File Structure After Training

```
bot 2/
├── go_emotions_dataset (1).csv
├── data_preprocessing.py
├── train_emotion_classifier.py
├── evaluate_model.py
├── inference.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── TECHNICAL_DOCS.md
├── QUICKSTART.md
│
├── processed_data/          ← Created after Step 1
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── label_mapping.json
│
└── models/                  ← Created after Step 2
    ├── best_model/
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   └── tokenizer files
    ├── best_model_metrics.json
    ├── test_results.json
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    ├── error_analysis.txt
    ├── evaluation_report.txt
    └── classification_report.txt
```

---

## 🔥 Pro Tips

1. **Use GPU**: Training is 60x faster on GPU
2. **Monitor Progress**: Watch validation F1 score
3. **Early Stopping**: If val loss increases, training stops
4. **Best Model**: Always saved based on highest val F1
5. **Test Diverse Examples**: Try sarcasm, mixed emotions

---

## 📞 Need Help?

1. Check `TECHNICAL_DOCS.md` for detailed explanations
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify dataset file exists and is readable

---

**Ready to start?** Run:
```bash
python run_pipeline.py
```

Good luck! 🎉
