"""
Tests for the NLP Sentiment Analysis Pipeline.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment_analyzer import (
    TextPreprocessor, LexiconAnalyzer, NaiveBayesSentiment,
    TfidfFeaturizer, SentimentEvaluator
)


class TestTextPreprocessor:
    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_basic_preprocessing(self):
        tokens = self.preprocessor.preprocess("Hello World! This is a TEST.")
        assert "hello" in tokens
        assert "world" in tokens

    def test_remove_stopwords(self):
        tokens = self.preprocessor.preprocess("the cat is on the mat", remove_stopwords=True)
        assert "the" not in tokens
        assert "cat" in tokens

    def test_empty_text(self):
        tokens = self.preprocessor.preprocess("")
        assert tokens == []


class TestLexiconAnalyzer:
    def setup_method(self):
        self.analyzer = LexiconAnalyzer()

    def test_positive(self):
        result = self.analyzer.analyze("This is excellent and amazing!")
        assert result["sentiment"] == "positive"
        assert result["compound"] > 0

    def test_negative(self):
        result = self.analyzer.analyze("This is terrible and awful.")
        assert result["sentiment"] == "negative"
        assert result["compound"] < 0

    def test_neutral(self):
        result = self.analyzer.analyze("The table has four legs.")
        assert result["sentiment"] == "neutral"

    def test_negation(self):
        result = self.analyzer.analyze("This is not good.")
        assert result["compound"] < 0.05

    def test_intensifier(self):
        base = self.analyzer.analyze("This is good.")
        intensified = self.analyzer.analyze("This is very good.")
        # Intensified should have different compound
        assert intensified["compound"] != base["compound"]

    def test_empty(self):
        result = self.analyzer.analyze("")
        assert result["sentiment"] == "neutral"


class TestNaiveBayesSentiment:
    def setup_method(self):
        self.classifier = NaiveBayesSentiment()
        self.train_texts = [
            "great product love it amazing",
            "excellent quality wonderful experience",
            "best purchase ever happy satisfied",
            "terrible product hate it awful",
            "horrible quality disappointing experience",
            "worst purchase ever angry frustrated",
        ]
        self.train_labels = ["positive", "positive", "positive",
                             "negative", "negative", "negative"]

    def test_fit_predict(self):
        self.classifier.fit(self.train_texts, self.train_labels)
        pred = self.classifier.predict("great quality love")
        assert pred == "positive"

    def test_predict_negative(self):
        self.classifier.fit(self.train_texts, self.train_labels)
        pred = self.classifier.predict("terrible awful horrible")
        assert pred == "negative"

    def test_predict_proba(self):
        self.classifier.fit(self.train_texts, self.train_labels)
        probs = self.classifier.predict_proba("great product")
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_not_fitted(self):
        with pytest.raises(RuntimeError):
            self.classifier.predict("test")


class TestTfidfFeaturizer:
    def test_fit_transform(self):
        featurizer = TfidfFeaturizer()
        texts = ["the cat sat", "the dog barked", "the cat barked"]
        vectors = featurizer.fit_transform(texts)
        assert len(vectors) == 3
        assert len(vectors[0]) == len(vectors[1])

    def test_transform(self):
        featurizer = TfidfFeaturizer()
        featurizer.fit_transform(["hello world", "test data"])
        vectors = featurizer.transform(["hello test"])
        assert len(vectors) == 1


class TestSentimentEvaluator:
    def test_accuracy(self):
        y_true = ["positive", "negative", "positive", "negative"]
        y_pred = ["positive", "negative", "positive", "negative"]
        assert SentimentEvaluator.accuracy(y_true, y_pred) == 1.0

    def test_accuracy_partial(self):
        y_true = ["positive", "negative", "positive", "negative"]
        y_pred = ["positive", "negative", "negative", "negative"]
        assert SentimentEvaluator.accuracy(y_true, y_pred) == 0.75

    def test_precision_recall_f1(self):
        y_true = ["positive", "positive", "negative", "negative"]
        y_pred = ["positive", "negative", "negative", "negative"]
        metrics = SentimentEvaluator.precision_recall_f1(y_true, y_pred)
        assert "positive" in metrics
        assert "negative" in metrics
        assert metrics["negative"]["recall"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
