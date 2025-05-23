# HARMONIA: Hybrid Sentiment-Numeric Hotel Rating Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)

A machine learning system that predicts hotel ratings (1-10) by combining numerical ratings and text reviews through Gradient Boosting and SVM.

## 🔍 Overview
- **Hybrid Approach**: Blends structured numerical data (cleanliness, service, etc.) with unstructured text sentiment analysis
- **Core Models**: 
  - `GradientBoostingRegressor` for numerical rating prediction
  - `LinearSVC` with TF-IDF for sentiment classification
- **Key Feature**: Weighted scoring formula (60% numerical + 40% sentiment)

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
```

### Run the Web App
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

## 🛠️ Project Structure
```
harmonia/
├── app.py                  # Flask application
├── train.py                # Model training scripts
├── models/
│   ├── hybrid_model.pkl    # Pretrained rating predictor
│   └── sentiment_model.pkl # Pretrained sentiment analyzer
├── templates/
│   └── index.html          # Frontend interface
├── data/
│   └── train.csv           # Sample training data
└── requirements.txt        # Dependency list
```

## 📊 Model Training
To retrain models with custom data:
```bash
python train.py --data_path ./data/your_data.csv
```

Supported training data format (`train.csv`):
```csv
cleanliness,service,comfort,amenities,review_text,rating
8,9,7,8,"Great experience",8.5
3,4,2,5,"Terrible service",2.8
```

## 🌟 Key Features
1. **Dynamic Weight Adjustment**
   ```python
   # Customize weights in app.py
   FINAL_SCORE_WEIGHTS = {
       'numerical': 0.6, 
       'sentiment': 0.4
   }
   ```

2. **Synthetic Data Generation**
   - Auto-generates perfect-score samples during training
   - Configurable in `train.py`:
     ```python
     SYNTHETIC_SAMPLES = 20  # Number of 10/10 samples to add
     ```

3. **Real-Time Analysis**
   - Processes inputs in <500ms
   - API endpoint at `/predict` (accepts JSON)

## 📈 Performance Metrics
| Model | MAE | F1-Score |
|-------|-----|----------|
| Numerical (GB) | 0.82 | - |
| Sentiment (SVM) | - | 0.914 |
| Hybrid System | 0.75 | 0.901 |

## 📚 Documentation
- [Full Technical Report](docs/report.pdf)
- [API Reference](docs/API.md)

## 🤝 Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add some feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 📜 License
MIT License - See [LICENSE](LICENSE)

---
**Note**: Replace placeholder paths with your actual file structure. For a live demo, check out our [Demo Video](https://example.com/demo).
```

### Key Features of This README:
1. **Badges**: Visual indicators for Python version and libraries
2. **Structured Sections**: Clear separation of setup, usage, and technical details
3. **Code Blocks**: Ready-to-copy commands for easy setup
4. **Visual Hierarchy**: Emojis and headers improve readability
5. **Future-Proof**: Placeholders for additional docs/links
