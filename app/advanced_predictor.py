"""
Advanced Ensemble Predictor for Casino Game Predictions

This module implements a sophisticated ensemble prediction system combining
XGBoost, LightGBM, and RandomForest models with advanced feature engineering
to achieve higher prediction accuracy.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging
from datetime import datetime
import os

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
        
        Features include:
        - Raw sequence values
        - Rolling statistics (mean, std, mode)
        - Streak information
        - Transition probabilities
        - Momentum indicators
        """
        features = []
        seq = np.array(sequence).flatten()
        
        # 1. Raw sequence values (last N values)
        features.extend(seq[-self.sequence_length:] if len(seq) >= self.sequence_length else np.pad(seq, (self.sequence_length - len(seq), 0)))
        
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
        
        # 4. Value distribution features
        for val in [0, 1, 2]:
            features.append(np.sum(seq == val) / len(seq) if len(seq) > 0 else 0)
        
        # 5. Transition probabilities
        if len(seq) >= 2:
            last_val = seq[-1]
            transitions = []
            for i in range(len(seq) - 1):
                if seq[i] == last_val:
                    transitions.append(seq[i + 1])
            if transitions:
                for val in [0, 1, 2]:
                    features.append(transitions.count(val) / len(transitions))
            else:
                features.extend([0.33, 0.33, 0.34])
        else:
            features.extend([0.33, 0.33, 0.34])
        
        # 6. Momentum indicators
        if len(seq) >= 5:
            recent_mean = np.mean(seq[-3:])
            older_mean = np.mean(seq[-6:-3])
            momentum = recent_mean - older_mean
            features.append(momentum)
        else:
            features.append(0)
        
        # 7. Alternation rate (how often value changes)
        if len(seq) >= 2:
            changes = sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])
            features.append(changes / (len(seq) - 1))
        else:
            features.append(0.5)
        
        # 8. Last N transition pattern
        if len(seq) >= 4:
            pattern = seq[-4:].astype(int)
            pattern_code = sum(v * (3 ** i) for i, v in enumerate(pattern))
            features.append(pattern_code / (3 ** 4))  # Normalized
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
        return streak, current_value
    
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
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1
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
        
        # Model weights (will be updated based on performance)
        self.model_weights = {
            'rf': 0.33,
            'xgb': 0.34,
            'lgb': 0.33
        }
        
        self.fitted = False
        self.classes_ = None
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
            X_seq, y = self.prepare_sequences(historical_data)
            if len(X_seq) < 10:
                logger.warning("Not enough sequences for training")
                return 0.0
            
            # Extract advanced features
            X = self.feature_engine.fit_transform(X_seq)
            
            # Store classes
            self.classes_ = np.unique(y)
            
            # Train individual models
            logger.info(f"Training ensemble for {self.endpoint_type} with {len(X)} samples")
            
            self.rf_model.fit(X, y)
            self.xgb_model.fit(X, y)
            self.lgb_model.fit(X, y)
            
            # Calculate individual model accuracies using cross-validation
            accuracies = {}
            for name, model in [('rf', self.rf_model), ('xgb', self.xgb_model), ('lgb', self.lgb_model)]:
                try:
                    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 10 + 1), scoring='accuracy')
                    accuracies[name] = np.mean(cv_scores)
                    logger.info(f"{name.upper()} CV accuracy: {accuracies[name]:.4f}")
                except Exception as e:
                    logger.warning(f"CV failed for {name}: {e}")
                    accuracies[name] = 0.5
            
            # Update model weights based on performance
            total_acc = sum(accuracies.values())
            if total_acc > 0:
                self.model_weights = {k: v / total_acc for k, v in accuracies.items()}
            
            self.fitted = True
            self.last_training_time = datetime.now()
            self.training_samples = len(X)
            
            # Calculate ensemble accuracy
            ensemble_accuracy = sum(acc * self.model_weights[name] for name, acc in accuracies.items())
            logger.info(f"Ensemble accuracy for {self.endpoint_type}: {ensemble_accuracy:.4f}")
            logger.info(f"Model weights: {self.model_weights}")
            
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
        
        # Extract features
        X = self.feature_engine.transform(sequence)
        
        # Get probabilities from each model
        rf_proba = self.rf_model.predict_proba(X)[0]
        xgb_proba = self.xgb_model.predict_proba(X)[0]
        lgb_proba = self.lgb_model.predict_proba(X)[0]
        
        # Ensure all probabilities have same shape
        n_classes = max(len(rf_proba), len(xgb_proba), len(lgb_proba))
        
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
        predicted_class = np.argmax(proba)
        confidence = proba[predicted_class]
        
        return int(predicted_class), float(confidence)
    
    def predict_with_strategy(self, sequence: np.ndarray, game_type: str) -> Tuple[str, float]:
        """
        Make a prediction with game-specific strategy.
        Applies anti-bias techniques and pattern analysis.
        """
        proba = self.predict_proba(sequence)
        
        if game_type in ['teen20', 'dt20']:
            # For teen20 and dt20, only values 1 and 2 are valid
            prob_1 = proba[1] if len(proba) > 1 else 0
            prob_2 = proba[2] if len(proba) > 2 else 0
            
            # Apply anti-bias correction
            # If recent results are heavily biased, counter-predict
            recent_bias = self._calculate_recent_bias(sequence, [1, 2])
            
            if recent_bias:
                biased_value, bias_strength = recent_bias
                if bias_strength > 0.3:  # Strong bias detected
                    # Reduce confidence in biased value
                    if biased_value == 1:
                        prob_1 *= (1 - bias_strength * 0.3)
                        prob_2 *= (1 + bias_strength * 0.2)
                    else:
                        prob_2 *= (1 - bias_strength * 0.3)
                        prob_1 *= (1 + bias_strength * 0.2)
            
            # Streak reversal logic
            streak_length, streak_value = self.feature_engine._get_current_streak(sequence)
            if streak_length >= 4:
                # High probability of streak ending
                if streak_value == 1:
                    prob_2 += 0.15
                else:
                    prob_1 += 0.15
            
            # Normalize
            total = prob_1 + prob_2
            if total > 0:
                prob_1 /= total
                prob_2 /= total
            
            # Final decision
            if prob_1 > prob_2:
                return "1", float(prob_1)
            else:
                return "2", float(prob_2)
        
        else:  # lucky7eu - values 0, 1, 2
            prob_0 = proba[0] if len(proba) > 0 else 0
            prob_1 = proba[1] if len(proba) > 1 else 0
            prob_2 = proba[2] if len(proba) > 2 else 0
            
            # Apply anti-bias correction
            recent_bias = self._calculate_recent_bias(sequence, [0, 1, 2])
            
            if recent_bias:
                biased_value, bias_strength = recent_bias
                if bias_strength > 0.25:
                    if biased_value == 0:
                        prob_0 *= (1 - bias_strength * 0.2)
                    elif biased_value == 1:
                        prob_1 *= (1 - bias_strength * 0.2)
                    else:
                        prob_2 *= (1 - bias_strength * 0.2)
            
            # Streak reversal
            streak_length, streak_value = self.feature_engine._get_current_streak(sequence)
            if streak_length >= 3:
                # Boost probability of other values
                if streak_value == 0:
                    prob_1 += 0.1
                    prob_2 += 0.1
                elif streak_value == 1:
                    prob_0 += 0.1
                    prob_2 += 0.1
                else:
                    prob_0 += 0.1
                    prob_1 += 0.1
            
            # Normalize
            total = prob_0 + prob_1 + prob_2
            if total > 0:
                prob_0 /= total
                prob_1 /= total
                prob_2 /= total
            
            # Final decision
            max_prob = max(prob_0, prob_1, prob_2)
            if max_prob == prob_0:
                return "0", float(prob_0)
            elif max_prob == prob_2:
                return "2", float(prob_2)
            else:
                return "1", float(prob_1)
    
    def _calculate_recent_bias(self, sequence: np.ndarray, valid_values: List[int]) -> Optional[Tuple[int, float]]:
        """Calculate if there's a strong bias in recent results."""
        recent = sequence[-10:] if len(sequence) >= 10 else sequence
        
        counts = Counter(recent)
        total = len(recent)
        
        if total == 0:
            return None
        
        max_count = max(counts.values())
        max_value = max(counts, key=counts.get)
        
        expected = total / len(valid_values)
        bias_strength = (max_count - expected) / total
        
        if bias_strength > 0.1:
            return max_value, bias_strength
        return None
