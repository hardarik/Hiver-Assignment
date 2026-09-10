import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

INTENTS = [
    "SHIPPING_DELIVERY_INQUIRY",
    "REFUND_RETURN_REQUEST",
    "DAMAGED_MISSING_ITEM",
    "ACCOUNT_SECURITY_BILLING",
    "DIGITAL_SERVICES_PRIME",
    "ESCALATION_HUMAN_COMPLAINT",
    "FEEDBACK_CHITCHAT_GENERAL"
]

class IntentClassifier:
    """
    Hybrid Intent Classifier combining TF-IDF N-gram feature representation with 
    Multinomial Naive Bayes and deterministic keyword boundary rules.
    """
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        self.is_fitted = False
        
        # Rule priority keywords for explicit domain signals
        self.rules = [
            (r'\b(pissed|terrible|worst|sucks|lawsuit|lawyer|bbb|scam|lied|useless|disappointed|disgusting|horrible)\b', "ESCALATION_HUMAN_COMPLAINT"),
            (r'\b(damaged|broken|empty box|missing item|opened|shattered|destroyed|tampered)\b', "DAMAGED_MISSING_ITEM"),
            (r'\b(refund|return|money back|reimburse|cancel order|cancellation)\b', "REFUND_RETURN_REQUEST"),
            (r'\b(account|sign in|login|password|locked|hacked|closed|credit card|charge|charged|billing)\b', "ACCOUNT_SECURITY_BILLING"),
            (r'\b(prime video|fire stick|kindle|buffering|stream|video error|playback|tv stick|app glitch)\b', "DIGITAL_SERVICES_PRIME"),
            (r'\b(where is my|tracking|delivered|not arrived|late|delay|shipping|courier|delivery|package)\b', "SHIPPING_DELIVERY_INQUIRY"),
        ]

    def fit(self, training_data: list):
        """Fit model on labeled training list of dicts or dataframe."""
        texts = [d['customer_text'] for d in training_data]
        labels = [d['ground_truth_intent'] for d in training_data]
        self.model.fit(texts, labels)
        self.is_fitted = True

    def predict(self, text: str) -> dict:
        """Predict primary intent with confidence and rule fallback."""
        text_lower = text.lower()

        # Check high-confidence rule overrides first for crisp safety boundaries
        for pattern, intent in self.rules:
            if re.search(pattern, text_lower):
                return {
                    "intent": intent,
                    "confidence": 0.95,
                    "method": "rule_assisted"
                }

        if self.is_fitted:
            probs = self.model.predict_proba([text])[0]
            classes = self.model.classes_
            max_idx = np.argmax(probs)
            return {
                "intent": classes[max_idx],
                "confidence": float(probs[max_idx]),
                "method": "tfidf_nb"
            }

        # Un-fitted fallback default
        return {
            "intent": "FEEDBACK_CHITCHAT_GENERAL",
            "confidence": 0.50,
            "method": "fallback"
        }
