"""
Sentiment Analysis Pipeline
Lexicon-based and Naive Bayes sentiment analysis with preprocessing and evaluation.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


class TextPreprocessor:
    """Text preprocessing for sentiment analysis."""

    STOP_WORDS = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "this", "that", "it", "its",
    })

    def preprocess(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """Clean and tokenize text."""
        text = text.lower()
        text = re.sub(r"[^a-z\s']", " ", text)
        tokens = text.split()
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOP_WORDS]
        return tokens


class LexiconAnalyzer:
    """VADER-inspired lexicon-based sentiment analyzer."""

    POSITIVE = {
        "good": 1.5, "great": 2.0, "excellent": 2.5, "amazing": 2.5,
        "wonderful": 2.5, "fantastic": 2.5, "love": 2.0, "best": 2.0,
        "happy": 1.5, "beautiful": 1.5, "nice": 1.0, "perfect": 2.5,
        "awesome": 2.0, "outstanding": 2.5, "brilliant": 2.0, "superb": 2.5,
        "enjoy": 1.5, "pleased": 1.5, "recommend": 1.5, "like": 1.0,
        "helpful": 1.5, "easy": 1.0, "fast": 1.0, "reliable": 1.5,
        "thank": 1.0, "well": 1.0, "better": 1.5, "effective": 1.5,
        "positive": 1.0, "incredible": 2.0, "delightful": 2.0, "impressive": 2.0,
    }

    NEGATIVE = {
        "bad": -1.5, "terrible": -2.5, "horrible": -2.5, "awful": -2.5,
        "worst": -2.5, "hate": -2.5, "poor": -1.5, "ugly": -1.5,
        "boring": -1.5, "annoying": -1.5, "disappointing": -2.0,
        "useless": -2.0, "broken": -1.5, "slow": -1.0, "difficult": -1.0,
        "problem": -1.5, "error": -1.5, "fail": -2.0, "failure": -2.0,
        "wrong": -1.5, "waste": -2.0, "never": -1.0, "unhappy": -1.5,
        "angry": -2.0, "frustrated": -2.0, "confusing": -1.5,
        "complicated": -1.0, "crash": -2.0, "bug": -1.5,
    }

    NEGATIONS = {"not", "no", "never", "neither", "nor", "don't", "doesn't",
                 "didn't", "won't", "wouldn't", "couldn't", "shouldn't",
                 "isn't", "aren't", "wasn't", "hardly", "barely", "scarcely"}

    INTENSIFIERS = {"very": 1.3, "really": 1.3, "extremely": 1.5,
                    "absolutely": 1.5, "totally": 1.4, "incredibly": 1.4,
                    "highly": 1.3, "so": 1.2, "quite": 1.1}

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def analyze(self, text: str) -> Dict:
        """Analyze sentiment using lexicon scoring."""
        tokens = self.preprocessor.preprocess(text)
        if not tokens:
            return {"sentiment": "neutral", "compound": 0.0,
                    "positive": 0.0, "negative": 0.0, "neutral": 1.0}

        scores = []
        for i, token in enumerate(tokens):
            score = self.POSITIVE.get(token, 0) + self.NEGATIVE.get(token, 0)
            if score != 0:
                if i > 0 and tokens[i - 1] in self.NEGATIONS:
                    score *= -0.75
                if i > 0 and tokens[i - 1] in self.INTENSIFIERS:
                    score *= self.INTENSIFIERS[tokens[i - 1]]
            scores.append(score)

        pos_sum = sum(s for s in scores if s > 0)
        neg_sum = sum(s for s in scores if s < 0)
        total = pos_sum + abs(neg_sum)
        compound = (pos_sum + neg_sum) / (total + 5.0) if total > 0 else 0.0
        compound = max(-1.0, min(1.0, compound))

        n = len(scores) if scores else 1
        pos_prop = sum(1 for s in scores if s > 0) / n
        neg_prop = sum(1 for s in scores if s < 0) / n

        label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"

        return {
            "sentiment": label,
            "compound": round(compound, 4),
            "positive": round(pos_prop, 4),
            "negative": round(neg_prop, 4),
            "neutral": round(1 - pos_prop - neg_prop, 4),
        }


class NaiveBayesSentiment:
    """Naive Bayes classifier for sentiment analysis."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_priors = {}
        self.feature_log_probs = {}
        self.vocabulary = set()
        self.classes = []
        self.preprocessor = TextPreprocessor()
        self._fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> "NaiveBayesSentiment":
        """Train on labeled text data."""
        class_counts = Counter(labels)
        self.classes = sorted(class_counts.keys())
        total = len(texts)

        self.class_log_priors = {
            c: math.log(n / total) for c, n in class_counts.items()
        }

        tokenized = [self.preprocessor.preprocess(t, remove_stopwords=True) for t in texts]
        self.vocabulary = set()
        for tokens in tokenized:
            self.vocabulary.update(tokens)

        vocab_size = len(self.vocabulary)
        class_word_counts = defaultdict(Counter)
        class_total = defaultdict(int)

        for tokens, label in zip(tokenized, labels):
            for token in tokens:
                class_word_counts[label][token] += 1
                class_total[label] += 1

        self.feature_log_probs = {}
        for cls in self.classes:
            self.feature_log_probs[cls] = {}
            for word in self.vocabulary:
                count = class_word_counts[cls][word]
                self.feature_log_probs[cls][word] = math.log(
                    (count + self.alpha) / (class_total[cls] + self.alpha * vocab_size)
                )

        self._fitted = True
        return self

    def predict(self, text: str) -> str:
        """Predict sentiment label."""
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        tokens = self.preprocessor.preprocess(text, remove_stopwords=True)
        log_probs = {}

        for cls in self.classes:
            lp = self.class_log_priors[cls]
            for token in tokens:
                if token in self.feature_log_probs[cls]:
                    lp += self.feature_log_probs[cls][token]
            log_probs[cls] = lp

        max_lp = max(log_probs.values())
        exp_sum = sum(math.exp(v - max_lp) for v in log_probs.values())
        return {c: math.exp(v - max_lp) / exp_sum for c, v in log_probs.items()}


class TfidfFeaturizer:
    """TF-IDF feature extraction for sentiment analysis."""

    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.preprocessor = TextPreprocessor()

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        """Fit on texts and return TF-IDF vectors."""
        tokenized = [self.preprocessor.preprocess(t, remove_stopwords=True) for t in texts]
        n_docs = len(texts)

        df = Counter()
        for tokens in tokenized:
            for token in set(tokens):
                df[token] += 1

        sorted_terms = sorted(df.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
        self.idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1
                    for term, freq in sorted_terms}

        return self._transform(tokenized)

    def transform(self, texts: List[str]) -> List[List[float]]:
        """Transform texts using fitted vocabulary."""
        tokenized = [self.preprocessor.preprocess(t, remove_stopwords=True) for t in texts]
        return self._transform(tokenized)

    def _transform(self, tokenized: List[List[str]]) -> List[List[float]]:
        vectors = []
        for tokens in tokenized:
            vec = [0.0] * len(self.vocabulary)
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            for term, idx in self.vocabulary.items():
                if term in tf:
                    vec[idx] = (tf[term] / total) * self.idf.get(term, 0)
            vectors.append(vec)
        return vectors


class SentimentEvaluator:
    """Evaluation metrics for sentiment analysis."""

    @staticmethod
    def accuracy(y_true: List[str], y_pred: List[str]) -> float:
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true) if y_true else 0.0

    @staticmethod
    def precision_recall_f1(y_true: List[str], y_pred: List[str]) -> Dict:
        classes = sorted(set(y_true + y_pred))
        metrics = {}

        for cls in classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            metrics[cls] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

        return metrics
