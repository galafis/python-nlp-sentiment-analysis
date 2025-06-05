"""
Core Sentiment Analyzer
Main class for sentiment analysis using multiple models
"""

import numpy as np
import logging
from typing import List, Dict, Union
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from preprocessing.text_cleaner import TextCleaner
from models.bert_model import BERTSentimentModel
from models.traditional_ml import TraditionalMLModel

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_type='bert', language='en'):
        """
        Initialize sentiment analyzer
        
        Args:
            model_type: Type of model to use ('bert', 'roberta', 'traditional', 'ensemble')
            language: Language for text processing ('en', 'pt', 'es')
        """
        self.model_type = model_type
        self.language = language
        self.text_cleaner = TextCleaner(language=language)
        
        # Initialize models based on type
        self._initialize_models()
        
        logger.info(f"Sentiment analyzer initialized with model: {model_type}")
    
    def _initialize_models(self):
        """Initialize the specified models"""
        try:
            if self.model_type == 'bert':
                self.model = self._load_bert_model()
            elif self.model_type == 'roberta':
                self.model = self._load_roberta_model()
            elif self.model_type == 'traditional':
                self.model = TraditionalMLModel()
            elif self.model_type == 'ensemble':
                self.models = {
                    'bert': self._load_bert_model(),
                    'traditional': TraditionalMLModel()
                }
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        except Exception as e:
            logger.warning(f"Error loading model {self.model_type}: {str(e)}")
            logger.info("Falling back to traditional ML model")
            self.model = TraditionalMLModel()
            self.model_type = 'traditional'
    
    def _load_bert_model(self):
        """Load BERT model for sentiment analysis"""
        try:
            # Try to load fine-tuned model first
            model_path = "models/trained/bert_sentiment"
            if torch.cuda.is_available():
                device = 0
            else:
                device = -1
            
            # Use Hugging Face pipeline for simplicity
            return pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=device,
                return_all_scores=True
            )
        except Exception as e:
            logger.warning(f"Error loading BERT model: {str(e)}")
            # Fallback to basic sentiment pipeline
            return pipeline("sentiment-analysis", return_all_scores=True)
    
    def _load_roberta_model(self):
        """Load RoBERTa model for sentiment analysis"""
        try:
            return pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
        except Exception as e:
            logger.warning(f"Error loading RoBERTa model: {str(e)}")
            return self._load_bert_model()
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment, confidence, and probabilities
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            }
        
        # Preprocess text
        cleaned_text = self.text_cleaner.preprocess_pipeline(text)
        
        if self.model_type == 'ensemble':
            return self._analyze_ensemble(cleaned_text)
        else:
            return self._analyze_single_model(cleaned_text)
    
    def _analyze_single_model(self, text: str) -> Dict:
        """Analyze using single model"""
        try:
            if self.model_type in ['bert', 'roberta']:
                return self._analyze_transformer(text)
            elif self.model_type == 'traditional':
                return self._analyze_traditional(text)
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            }
    
    def _analyze_transformer(self, text: str) -> Dict:
        """Analyze using transformer model"""
        try:
            # Get predictions
            results = self.model(text)
            
            # Process results
            if isinstance(results[0], list):
                scores = results[0]
            else:
                scores = results
            
            # Convert to standard format
            probabilities = {}
            max_score = 0
            predicted_sentiment = 'neutral'
            
            for score in scores:
                label = score['label'].lower()
                prob = score['score']
                
                # Map labels to standard format
                if 'pos' in label or label == 'label_2':
                    probabilities['positive'] = prob
                    if prob > max_score:
                        max_score = prob
                        predicted_sentiment = 'positive'
                elif 'neg' in label or label == 'label_0':
                    probabilities['negative'] = prob
                    if prob > max_score:
                        max_score = prob
                        predicted_sentiment = 'negative'
                else:  # neutral or label_1
                    probabilities['neutral'] = prob
                    if prob > max_score:
                        max_score = prob
                        predicted_sentiment = 'neutral'
            
            # Ensure all probabilities are present
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment not in probabilities:
                    probabilities[sentiment] = 0.0
            
            return {
                'sentiment': predicted_sentiment,
                'confidence': max_score,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"Error in transformer analysis: {str(e)}")
            return self._get_default_result()
    
    def _analyze_traditional(self, text: str) -> Dict:
        """Analyze using traditional ML model"""
        try:
            return self.model.predict(text)
        except Exception as e:
            logger.error(f"Error in traditional ML analysis: {str(e)}")
            return self._get_default_result()
    
    def _analyze_ensemble(self, text: str) -> Dict:
        """Analyze using ensemble of models"""
        try:
            results = []
            
            # Get predictions from all models
            for model_name, model in self.models.items():
                if model_name in ['bert', 'roberta']:
                    result = self._analyze_transformer(text)
                else:
                    result = model.predict(text)
                results.append(result)
            
            # Ensemble predictions (simple averaging)
            ensemble_probs = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            for result in results:
                for sentiment, prob in result['probabilities'].items():
                    ensemble_probs[sentiment] += prob
            
            # Average probabilities
            num_models = len(results)
            for sentiment in ensemble_probs:
                ensemble_probs[sentiment] /= num_models
            
            # Get final prediction
            predicted_sentiment = max(ensemble_probs, key=ensemble_probs.get)
            confidence = ensemble_probs[predicted_sentiment]
            
            return {
                'sentiment': predicted_sentiment,
                'confidence': confidence,
                'probabilities': ensemble_probs
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble analysis: {str(e)}")
            return self._get_default_result()
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        logger.info(f"Analyzing {len(texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                result = self.analyze(text)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if i + batch_size < len(texts):
                logger.info(f"Processed {i + batch_size}/{len(texts)} texts")
        
        logger.info(f"Completed analysis of {len(texts)} texts")
        return results
    
    def _get_default_result(self) -> Dict:
        """Get default result for error cases"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            'model_type': self.model_type,
            'language': self.language,
            'supports_batch': True,
            'supports_probabilities': True
        }

