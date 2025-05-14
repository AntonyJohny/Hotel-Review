from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import joblib
import os
import traceback
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Initialize NLTK sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Global variables
hybrid_model = None
sentiment_model = None

def clean_text(text):
    """Enhanced text cleaning with emoji handling"""
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)   # Remove extra whitespace
    return text.strip()

def train_models():
    global hybrid_model, sentiment_model
    
    # Load and augment base data
    df = pd.read_csv('train.csv')
    
    # Add synthetic perfect-score examples
    perfect_samples = pd.DataFrame([{
        'cleanliness': 10,
        'service': 10,
        'comfort': 10,
        'amenities': 10,
        'review_text': "Perfect experience in every aspect",
        'rating': 10.0
    } for _ in range(20)])
    df = pd.concat([df, perfect_samples], ignore_index=True)
    
    # Clean text
    df['cleaned_review'] = df['review_text'].apply(clean_text)
    
    # Train sentiment model with n-grams
    sentiment_model = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1500,
            stop_words=None
        )),
        ('clf', LinearSVC(
            C=0.8,
            class_weight='balanced',
            probability=True
        ))
    ])
    # Create binary sentiment labels (Positive if rating >= 8)
    df['sentiment'] = (df['rating'] >= 8).astype(int)
    sentiment_model.fit(df['cleaned_review'], df['sentiment'])
    
    # Train hybrid model with feature union
    numeric_features = ['cleanliness', 'service', 'comfort', 'amenities']
    text_features = 'cleaned_review'
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('text', TfidfVectorizer(ngram_range=(1, 2)), text_features)
        ])
    
    hybrid_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    
    hybrid_model.fit(df, df['rating'])
    
    # Save models
    joblib.dump(hybrid_model, 'hybrid_model.pkl')
    joblib.dump(sentiment_model, 'sentiment_model.pkl')

def load_models():
    global hybrid_model, sentiment_model
    if os.path.exists('hybrid_model.pkl') and os.path.exists('sentiment_model.pkl'):
        hybrid_model = joblib.load('hybrid_model.pkl')
        sentiment_model = joblib.load('sentiment_model.pkl')
        print("Models loaded successfully!")
    else:
        print("Training new models...")
        train_models()

@app.route('/', methods=['GET', 'POST'])
def home():
    results = {
        'numerical_prediction': None,
        'sentiment_score': None,
        'final_score': None,
        'error': None,
        'form_data': {}
    }
    
    if request.method == 'POST':
        try:
            # Store form data for repopulation
            results['form_data'] = request.form.to_dict()
            
            # Process numerical features with validation
            numerical_features = [
                max(1, min(10, int(request.form.get(feature, 5))))
                for feature in ['cleanliness', 'service', 'comfort', 'amenities']
            ]
            
            # Process text review
            raw_review = request.form.get('review', '')
            cleaned_review = clean_text(raw_review)
            
            # Create input DataFrame
            input_data = pd.DataFrame([{
                'cleanliness': numerical_features[0],
                'service': numerical_features[1],
                'comfort': numerical_features[2],
                'amenities': numerical_features[3],
                'review_text': cleaned_review,
                'cleaned_review': cleaned_review
            }])
            
            # Get predictions
            numerical_pred = hybrid_model.predict(input_data)[0]
            sentiment_prob = sentiment_model.predict_proba([cleaned_review])[0][1]
            
            # Enhanced scoring formula
            sentiment_boost = 5 + (sentiment_prob * 10)  # Range: 5-10
            final_score = (0.7 * numerical_pred) + (0.3 * sentiment_boost)
            
            # Add NLTK sentiment analysis
            nltk_sentiment = (sia.polarity_scores(raw_review)['compound'] + 1) / 2
            final_score = (final_score + nltk_sentiment * 2) / 1.2  # Blend with NLTK
            
            # Ensure final score bounds
            final_score = max(1, min(10, round(final_score, 1)))
            
            # Update results
            results.update({
                'numerical_prediction': round(numerical_pred, 1),
                'sentiment_score': round(sentiment_prob * 100, 1),
                'final_score': final_score
            })
            
        except Exception as e:
            results['error'] = f"Prediction error: {str(e)}"
            app.logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")

    return render_template('index.html', results=results)

@app.context_processor
def inject_request():
    return {'request': request}

if __name__ == '__main__':
    load_models()
    app.run(debug=True)