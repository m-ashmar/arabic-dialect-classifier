from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DialectClassifier:
    """Arabic dialect classification model"""
    
    def __init__(self):
        self.pipeline = make_pipeline(
            TfidfVectorizer(),
            StandardScaler(with_mean=False),
            SMOTE(random_state=42),
            MultinomialNB()
        )

    def train(self, X_train, y_train):
        """Train the model"""
        if X_train.empty or y_train.empty:
            raise ValueError("Training data is empty")

        # Clean labels
        y_train = y_train.astype('category').cat.remove_unused_categories()
        valid_mask = y_train.notna() & (y_train != 'unknown')
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        # Hyperparameter tuning
        param_grid = {
            'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
            'multinomialnb__alpha': [0.5, 0.8, 1.0]
        }

        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        self.pipeline = grid_search.best_estimator_
        logger.info(f"أفضل المعلمات: {grid_search.best_params_}")

    def evaluate(self, X_test, y_true):
        """Evaluate model performance"""
        y_pred = self.pipeline.predict(X_test)
        y_true = y_true.astype('category').cat.remove_unused_categories()
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }