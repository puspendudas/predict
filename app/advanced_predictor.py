"""
Advanced Ensemble Predictor for Casino Game Predictions

This module implements a sophisticated ensemble prediction system combining
XGBoost, LightGBM, and RandomForest models with advanced feature engineering
to achieve higher prediction accuracy.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging
from datetime import datetime
import os
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class AdvancedFeatureEngine:
    """
    Advanced feature engineering for time-series prediction.
    Extracts statistical, pattern-based, and temporal features.
    """
    
    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.fitted = False
    
    def extract_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive features from a sequence of results.
        """
        features = []
        seq = np.array(sequence).flatten()
        
        # 1. Raw sequence values (last N values)
        if len(seq) >= self.sequence_length:
            features.extend(seq[-self.sequence_length:])
        else:
            features.extend(np.pad(seq, (self.sequence_length - len(seq), 0)))
        
        # 2. Rolling statistics
        for window in [3, 5, 10]:
            if len(seq) >= window:
                window_data = seq[-window:]
                features.append(np.mean(window_data))
                features.append(np.std(window_data))
                features.append(Counter(window_data).most_common(1)[0][0])  # Mode
            else:
                features.extend([0, 0, 0])
        
        # 3. Streak detection
        streak_length, streak_value = self._get_current_streak(seq)
        features.append(streak_length)
        features.append(streak_value)
        
        # 4. Value distribution features (normalized for any class range)
        unique_vals = np.unique(seq)
        for i in range(3):  # Up to 3 distribution features
            if i < len(unique_vals):
                features.append(np.sum(seq == unique_vals[i]) / len(seq) if len(seq) > 0 else 0)
            else:
                features.append(0)
        
        # 5. Transition probabilities
        if len(seq) >= 2:
            last_val = seq[-1]
            transitions = []
            for i in range(len(seq) - 1):
                if seq[i] == last_val:
                    transitions.append(seq[i + 1])
            if transitions:
                unique_trans = np.unique(transitions)
                for i in range(3):
                    if i < len(unique_trans):
                        features.append(transitions.count(unique_trans[i]) / len(transitions))
                    else:
                        features.append(0.33)
            else:
                features.extend([0.33, 0.33, 0.34])
        else:
            features.extend([0.33, 0.33, 0.34])
        
        # 6. Momentum indicators
        if len(seq) >= 5:
            recent_mean = np.mean(seq[-3:])
            older_mean = np.mean(seq[-6:-3]) if len(seq) >= 6 else np.mean(seq[:3])
            momentum = recent_mean - older_mean
            features.append(momentum)
        else:
            features.append(0)
        
        # 7. Alternation rate
        if len(seq) >= 2:
            changes = sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])
            features.append(changes / (len(seq) - 1))
        else:
            features.append(0.5)
        
        # 8. Last N transition pattern
        if len(seq) >= 4:
            pattern = seq[-4:].astype(int)
            pattern_code = sum(v * (4 ** i) for i, v in enumerate(pattern))
            features.append(pattern_code / (4 ** 4))  # Normalized
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float64)
    
    def _get_current_streak(self, seq: np.ndarray) -> Tuple[int, int]:
        """Get the current streak length and value."""
        if len(seq) == 0:
            return 0, 0
        
        current_value = seq[-1]
        streak = 1
        for i in range(len(seq) - 2, -1, -1):
            if seq[i] == current_value:
                streak += 1
            else:
                break
        return streak, int(current_value)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the scaler and transform features."""
        features_list = [self.extract_features(x) for x in X]
        features_array = np.array(features_list)
        self.fitted = True
        return self.scaler.fit_transform(features_array)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        features_list = [self.extract_features(x) for x in X]
        features_array = np.array(features_list)
        if self.fitted:
            return self.scaler.transform(features_array)
        return features_array


class AdvancedEnsemblePredictor:
    """
    Advanced ensemble predictor combining multiple ML models
    with weighted voting based on recent performance.
    """
    
    def __init__(self, endpoint_type: str):
        self.endpoint_type = endpoint_type
        self.sequence_length = int(os.getenv("SEQUENCE_LENGTH", "20"))
        self.feature_engine = AdvancedFeatureEngine(self.sequence_length)
        self.label_encoder = LabelEncoder()
        
        # Initialize individual models with optimized hyperparameters
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # XGBoost without deprecated use_label_encoder
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1,
            verbosity=0
        )
        
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        # Model weights
        self.model_weights = {
            'rf': 0.33,
            'xgb': 0.34,
            'lgb': 0.33
        }
        
        self.fitted = False
        self.classes_ = None
        self.original_classes_ = None  # Store original class labels
        self.last_training_time = None
        self.training_samples = 0
    
    def prepare_sequences(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training sequences from historical data."""
        results = [int(d["result"]) for d in data]
        X, y = [], []
        
        for i in range(len(results) - self.sequence_length):
            X.append(results[i:i + self.sequence_length])
            y.append(results[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def fit(self, historical_data: List[Dict]) -> float:
        """
        Train the ensemble on historical data.
        Returns the estimated accuracy.
        """
        min_samples = int(os.getenv("MIN_TRAINING_SAMPLES", "50"))
        
        if len(historical_data) < min_samples:
            logger.warning(f"Insufficient data for training: {len(historical_data)} samples")
            return 0.0
        
        try:
            # Prepare sequences
            X_seq, y_original = self.prepare_sequences(historical_data)
            if len(X_seq) < 10:
                logger.warning("Not enough sequences for training")
                return 0.0
            
            # Store original classes and encode labels to 0, 1, 2, ...
            self.original_classes_ = np.unique(y_original)
            self.label_encoder.fit(y_original)
            y = self.label_encoder.transform(y_original)
            
            # Extract advanced features
            X = self.feature_engine.fit_transform(X_seq)
            
            # Store encoded classes
            self.classes_ = np.unique(y)
            
            # Train individual models
            logger.info(f"Training ensemble for {self.endpoint_type} with {len(X)} samples")
            logger.info(f"Original classes: {self.original_classes_}, Encoded classes: {self.classes_}")
            
            self.rf_model.fit(X, y)
            self.xgb_model.fit(X, y)
            self.lgb_model.fit(X, y)
            
            # Calculate individual model accuracies using cross-validation
            accuracies = {}
            for name, model in [('rf', self.rf_model), ('xgb', self.xgb_model), ('lgb', self.lgb_model)]:
                try:
                    n_splits = min(5, max(2, len(X) // 20))
                    cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring='accuracy')
                    accuracies[name] = np.mean(cv_scores)
                    logger.info(f"{name.upper()} CV accuracy: {accuracies[name]:.4f}")
                except Exception as e:
                    logger.warning(f"CV failed for {name}: {e}")
                    accuracies[name] = 0.5
            
            # Update model weights based on performance
            total_acc = sum(accuracies.values())
            if total_acc > 0:
                self.model_weights = {k: float(v / total_acc) for k, v in accuracies.items()}
            
            self.fitted = True
            self.last_training_time = datetime.now()
            self.training_samples = len(X)
            
            # Calculate ensemble accuracy
            ensemble_accuracy = sum(acc * self.model_weights[name] for name, acc in accuracies.items())
            logger.info(f"Ensemble accuracy for {self.endpoint_type}: {ensemble_accuracy:.4f}")
            
            return ensemble_accuracy
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return 0.0
    
    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities for a sequence using weighted ensemble.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        # Reshape if needed
        if sequence.ndim == 1:
            sequence = sequence.reshape(1, -1)
        
        # Extract features (returns numpy array, no feature names)
        X = self.feature_engine.transform(sequence)
        
        # Get probabilities from each model
        rf_proba = self.rf_model.predict_proba(X)[0]
        xgb_proba = self.xgb_model.predict_proba(X)[0]
        lgb_proba = self.lgb_model.predict_proba(X)[0]
        
        # Ensure all probabilities have same shape
        n_classes = len(self.classes_)
        
        def pad_proba(proba, target_size):
            if len(proba) < target_size:
                padded = np.zeros(target_size)
                padded[:len(proba)] = proba
                return padded
            return proba
        
        rf_proba = pad_proba(rf_proba, n_classes)
        xgb_proba = pad_proba(xgb_proba, n_classes)
        lgb_proba = pad_proba(lgb_proba, n_classes)
        
        # Weighted average of probabilities
        ensemble_proba = (
            self.model_weights['rf'] * rf_proba +
            self.model_weights['xgb'] * xgb_proba +
            self.model_weights['lgb'] * lgb_proba
        )
        
        return ensemble_proba
    
    def predict(self, sequence: np.ndarray) -> Tuple[int, float]:
        """
        Make a prediction with confidence score.
        Returns (predicted_class, confidence).
        """
        proba = self.predict_proba(sequence)
        predicted_idx = np.argmax(proba)
        confidence = proba[predicted_idx]
        
        # Convert back to original class label
        predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        return int(predicted_class), float(confidence)
    
    def predict_with_strategy(self, sequence: np.ndarray, game_type: str) -> Tuple[str, float]:
        """
        Make a prediction with game-specific strategy.
        Applies anti-bias techniques and pattern analysis.
        """
        proba = self.predict_proba(sequence)
        n_classes = len(self.classes_)
        
        # Map probabilities to original class labels
        class_probs = {}
        for i, orig_class in enumerate(self.original_classes_):
            if i < len(proba):
                class_probs[int(orig_class)] = float(proba[i])
            else:
                class_probs[int(orig_class)] = 0.0
        
        if game_type in ['teen20', 'dt20']:
            # For teen20 and dt20, values are 1 and 2 (or 1, 2, 3 for dt20)
            valid_values = [v for v in class_probs.keys() if v in [1, 2, 3]]
            if not valid_values:
                valid_values = list(class_probs.keys())
            
            # Get probabilities for valid values
            probs = {v: class_probs.get(v, 0) for v in valid_values}
            
            # Apply anti-bias correction
            recent_bias = self._calculate_recent_bias(sequence, valid_values)
            
            if recent_bias:
                biased_value, bias_strength = recent_bias
                if bias_strength > 0.3:
                    if biased_value in probs:
                        probs[biased_value] *= (1 - bias_strength * 0.3)
            
            # Streak reversal logic
            streak_length, streak_value = self.feature_engine._get_current_streak(sequence)
            if streak_length >= 4:
                for v in probs:
                    if v != streak_value:
                        probs[v] += 0.1
            
            # Normalize
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
            
            # Get best prediction
            best_value = max(probs, key=probs.get)
            return str(best_value), float(probs[best_value])
        
        else:  # lucky7eu - values 0, 1, 2
            valid_values = [0, 1, 2]
            probs = {v: class_probs.get(v, 0.33) for v in valid_values}
            
            # Apply anti-bias correction
            recent_bias = self._calculate_recent_bias(sequence, valid_values)
            
            if recent_bias:
                biased_value, bias_strength = recent_bias
                if bias_strength > 0.25:
                    if biased_value in probs:
                        probs[biased_value] *= (1 - bias_strength * 0.2)
            
            # Streak reversal
            streak_length, streak_value = self.feature_engine._get_current_streak(sequence)
            if streak_length >= 3:
                for v in probs:
                    if v != streak_value:
                        probs[v] += 0.1
            
            # Normalize
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
            
            # Get best prediction
            best_value = max(probs, key=probs.get)
            return str(best_value), float(probs[best_value])
    
    def _calculate_recent_bias(self, sequence: np.ndarray, valid_values: List[int]) -> Optional[Tuple[int, float]]:
        """Calculate if there's a strong bias in recent results."""
        recent = sequence[-10:] if len(sequence) >= 10 else sequence
        
        counts = Counter(recent)
        total = len(recent)
        
        if total == 0:
            return None
        
        max_count = max(counts.values())
        max_value = int(max(counts, key=counts.get))
        
        expected = total / len(valid_values)
        bias_strength = (max_count - expected) / total
        
        if bias_strength > 0.1:
            return max_value, bias_strength
        return None
