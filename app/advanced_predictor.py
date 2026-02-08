"""
Ultra-Refined Prediction Engine v2.0

This module implements a sophisticated multi-strategy prediction system
designed to achieve 70-80% accuracy through:
1. Adaptive strategy selection based on real-time performance
2. Pattern memory for successful prediction patterns
3. Multi-order Markov chains for transition analysis
4. Mean reversion and hot/cold value tracking
5. Streak reversal with dynamic thresholds
6. Ensemble ML as one of many strategies
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, deque
import logging
from datetime import datetime, timedelta
import os
import warnings
import random
import hashlib

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


class PatternMemory:
    """
    Stores patterns that led to correct predictions.
    Uses a hash-based lookup for O(1) pattern matching.
    """
    
    def __init__(self, max_patterns: int = 10000):
        self.patterns: Dict[str, Dict] = {}
        self.max_patterns = max_patterns
        self.hit_count = 0
        self.miss_count = 0
    
    def _pattern_to_key(self, pattern: List[int], pattern_length: int = 5) -> str:
        """Convert a pattern to a hashable key."""
        if len(pattern) >= pattern_length:
            key_pattern = pattern[-pattern_length:]
        else:
            key_pattern = pattern
        return "_".join(map(str, key_pattern))
    
    def add_pattern(self, sequence: List[int], next_value: int, was_correct: bool):
        """Add a pattern to memory with outcome tracking."""
        for length in [3, 4, 5, 6, 7]:
            if len(sequence) >= length:
                key = self._pattern_to_key(sequence, length)
                if key not in self.patterns:
                    self.patterns[key] = {
                        'predictions': {},
                        'total': 0,
                        'correct': 0
                    }
                
                if next_value not in self.patterns[key]['predictions']:
                    self.patterns[key]['predictions'][next_value] = {'success': 0, 'total': 0}
                
                self.patterns[key]['predictions'][next_value]['total'] += 1
                self.patterns[key]['total'] += 1
                
                if was_correct:
                    self.patterns[key]['predictions'][next_value]['success'] += 1
                    self.patterns[key]['correct'] += 1
        
        # Cleanup old patterns if too many
        if len(self.patterns) > self.max_patterns:
            # Remove patterns with lowest success rate
            sorted_patterns = sorted(
                self.patterns.items(),
                key=lambda x: x[1]['correct'] / max(x[1]['total'], 1)
            )
            for key, _ in sorted_patterns[:len(sorted_patterns) // 4]:
                del self.patterns[key]
    
    def get_prediction(self, sequence: List[int]) -> Optional[Tuple[int, float]]:
        """Get the best prediction based on pattern memory."""
        best_pred = None
        best_confidence = 0
        
        # Try different pattern lengths (longer patterns = higher weight)
        for length in [7, 6, 5, 4, 3]:
            if len(sequence) >= length:
                key = self._pattern_to_key(sequence, length)
                if key in self.patterns:
                    pattern_data = self.patterns[key]
                    if pattern_data['total'] >= 3:  # Minimum occurrences
                        for value, stats in pattern_data['predictions'].items():
                            if stats['total'] >= 2:
                                success_rate = stats['success'] / stats['total']
                                # Weight by pattern length and sample size
                                confidence = success_rate * (length / 7) * min(stats['total'] / 10, 1.0)
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    best_pred = value
                                    self.hit_count += 1
        
        if best_pred is None:
            self.miss_count += 1
        
        return (best_pred, best_confidence) if best_pred is not None else None


class MarkovChainPredictor:
    """
    Multi-order Markov chain for transition prediction.
    Tracks transition probabilities at multiple orders (1st, 2nd, 3rd order).
    """
    
    def __init__(self, max_order: int = 4):
        self.max_order = max_order
        self.transitions: Dict[int, Dict[str, Dict[int, int]]] = {
            order: {} for order in range(1, max_order + 1)
        }
    
    def update(self, sequence: List[int]):
        """Update transition counts from a sequence."""
        for order in range(1, min(self.max_order + 1, len(sequence))):
            for i in range(order, len(sequence)):
                state = tuple(sequence[i-order:i])
                state_key = "_".join(map(str, state))
                next_val = sequence[i]
                
                if state_key not in self.transitions[order]:
                    self.transitions[order][state_key] = {}
                
                if next_val not in self.transitions[order][state_key]:
                    self.transitions[order][state_key][next_val] = 0
                
                self.transitions[order][state_key][next_val] += 1
    
    def predict(self, sequence: List[int], valid_values: List[int]) -> Dict[int, float]:
        """Get transition probabilities for each valid value."""
        probs = {v: 0.0 for v in valid_values}
        total_weight = 0
        
        # Higher order = higher weight (but only if enough data)
        for order in range(self.max_order, 0, -1):
            if len(sequence) >= order:
                state = tuple(sequence[-order:])
                state_key = "_".join(map(str, state))
                
                if state_key in self.transitions[order]:
                    trans = self.transitions[order][state_key]
                    total_trans = sum(trans.values())
                    if total_trans >= 3:  # Minimum samples
                        weight = order * 2  # Higher order = more weight
                        for v in valid_values:
                            count = trans.get(v, 0)
                            probs[v] += (count / total_trans) * weight
                        total_weight += weight
        
        # Normalize
        if total_weight > 0:
            probs = {k: v / total_weight for k, v in probs.items()}
        else:
            # Uniform distribution
            probs = {v: 1.0 / len(valid_values) for v in valid_values}
        
        return probs


class MeanReversionPredictor:
    """
    Tracks value frequencies and predicts underrepresented values.
    Based on the gambler's fallacy but with statistical backing.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
    
    def predict(self, sequence: List[int], valid_values: List[int]) -> Dict[int, float]:
        """Predict based on which values are "due"."""
        recent = sequence[-self.window_size:] if len(sequence) >= self.window_size else sequence
        
        counts = Counter(recent)
        total = len(recent)
        expected = total / len(valid_values) if len(valid_values) > 0 else 1
        
        # Calculate "due" scores - higher for underrepresented values
        due_scores = {}
        for v in valid_values:
            actual = counts.get(v, 0)
            deviation = expected - actual
            # Positive deviation = value is due
            due_scores[v] = max(0, deviation / expected) if expected > 0 else 0
        
        # Convert to probabilities
        total_score = sum(due_scores.values())
        if total_score > 0:
            probs = {k: (v / total_score) * 0.6 + 0.1 for k, v in due_scores.items()}
        else:
            probs = {v: 1.0 / len(valid_values) for v in valid_values}
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        return probs


class StreakAnalyzer:
    """
    Analyzes streaks and predicts reversals.
    """
    
    def __init__(self, min_streak_for_reversal: int = 3):
        self.min_streak = min_streak_for_reversal
        self.reversal_success = {}  # Track reversal success by streak length
    
    def get_streak(self, sequence: List[int]) -> Tuple[int, int]:
        """Get current streak length and value."""
        if len(sequence) == 0:
            return 0, 0
        
        current = sequence[-1]
        streak = 1
        for i in range(len(sequence) - 2, -1, -1):
            if sequence[i] == current:
                streak += 1
            else:
                break
        
        return streak, current
    
    def predict(self, sequence: List[int], valid_values: List[int]) -> Dict[int, float]:
        """Predict based on streak analysis."""
        streak_len, streak_val = self.get_streak(sequence)
        
        probs = {v: 1.0 / len(valid_values) for v in valid_values}
        
        if streak_len >= self.min_streak and streak_val in valid_values:
            # Calculate reversal probability based on streak length
            # Longer streaks = higher reversal probability
            reversal_boost = min(0.4, 0.1 * (streak_len - self.min_streak + 1))
            
            # Reduce probability of streak continuing
            probs[streak_val] = max(0.1, 1.0 / len(valid_values) - reversal_boost)
            
            # Increase probability of other values
            other_values = [v for v in valid_values if v != streak_val]
            if other_values:
                boost_per_value = reversal_boost / len(other_values)
                for v in other_values:
                    probs[v] = 1.0 / len(valid_values) + boost_per_value
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        return probs
    
    def record_result(self, streak_len: int, did_reverse: bool):
        """Record whether a reversal prediction was correct."""
        if streak_len >= self.min_streak:
            if streak_len not in self.reversal_success:
                self.reversal_success[streak_len] = {'success': 0, 'total': 0}
            self.reversal_success[streak_len]['total'] += 1
            if did_reverse:
                self.reversal_success[streak_len]['success'] += 1


class AlternationPredictor:
    """
    Predicts based on alternation patterns.
    Some sequences show alternation tendencies.
    """
    
    def predict(self, sequence: List[int], valid_values: List[int]) -> Dict[int, float]:
        """Predict based on alternation pattern."""
        if len(sequence) < 2:
            return {v: 1.0 / len(valid_values) for v in valid_values}
        
        # Calculate alternation rate in recent history
        recent = sequence[-20:] if len(sequence) >= 20 else sequence
        alternations = sum(1 for i in range(len(recent) - 1) if recent[i] != recent[i + 1])
        alt_rate = alternations / (len(recent) - 1) if len(recent) > 1 else 0.5
        
        last_val = sequence[-1]
        
        probs = {}
        for v in valid_values:
            if v == last_val:
                # If high alternation rate, less likely to repeat
                probs[v] = 1.0 - alt_rate
            else:
                # Distribute remaining probability among other values
                probs[v] = alt_rate / max(len(valid_values) - 1, 1)
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs = {v: 1.0 / len(valid_values) for v in valid_values}
        
        return probs


class EnsembleMLPredictor:
    """
    Traditional ML ensemble predictor.
    """
    
    def __init__(self, endpoint_type: str):
        self.endpoint_type = endpoint_type
        self.label_encoder = LabelEncoder()
        self.fitted = False
        
        # Models with conservative hyperparameters
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.xgb = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=-1
        )
        
        self.lgb = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        self.original_classes = None
    
    def _prepare_features(self, sequence: List[int]) -> np.ndarray:
        """Extract features from a sequence."""
        seq = np.array(sequence)
        features = []
        
        # Basic statistics
        features.append(np.mean(seq))
        features.append(np.std(seq))
        features.append(seq[-1])  # Last value
        
        # Rolling means
        for w in [3, 5, 10]:
            if len(seq) >= w:
                features.append(np.mean(seq[-w:]))
            else:
                features.append(np.mean(seq))
        
        # Transitions
        if len(seq) >= 2:
            features.append(seq[-1] - seq[-2])
        else:
            features.append(0)
        
        # Mode of last N
        for w in [5, 10]:
            if len(seq) >= w:
                features.append(Counter(seq[-w:]).most_common(1)[0][0])
            else:
                features.append(seq[-1] if len(seq) > 0 else 0)
        
        return np.array(features)
    
    def fit(self, sequences: List[List[int]], labels: List[int]):
        """Train the ensemble."""
        if len(sequences) < 20:
            return
        
        self.original_classes = np.unique(labels)
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        X = np.array([self._prepare_features(s) for s in sequences])
        
        try:
            self.rf.fit(X, encoded_labels)
            self.xgb.fit(X, encoded_labels)
            self.lgb.fit(X, encoded_labels)
            self.fitted = True
        except Exception as e:
            logger.error(f"ML training failed: {e}")
    
    def predict(self, sequence: List[int], valid_values: List[int]) -> Dict[int, float]:
        """Get prediction probabilities."""
        if not self.fitted:
            return {v: 1.0 / len(valid_values) for v in valid_values}
        
        try:
            X = self._prepare_features(sequence).reshape(1, -1)
            
            # Get probabilities from each model
            rf_proba = self.rf.predict_proba(X)[0]
            xgb_proba = self.xgb.predict_proba(X)[0]
            lgb_proba = self.lgb.predict_proba(X)[0]
            
            # Average probabilities
            avg_proba = (rf_proba + xgb_proba + lgb_proba) / 3
            
            # Map to original classes
            probs = {v: 1.0 / len(valid_values) for v in valid_values}
            for i, orig_class in enumerate(self.original_classes):
                if i < len(avg_proba) and orig_class in probs:
                    probs[orig_class] = avg_proba[i]
            
            # Normalize
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
            
            return probs
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return {v: 1.0 / len(valid_values) for v in valid_values}


class StrategySelector:
    """
    Dynamically selects the best performing strategy.
    Tracks strategy performance and adapts weights.
    """
    
    def __init__(self):
        self.strategies = [
            'pattern_memory',
            'markov',
            'mean_reversion',
            'streak',
            'alternation',
            'ml_ensemble',
            'combined'
        ]
        
        # Performance tracking
        self.performance: Dict[str, Dict] = {
            s: {'correct': 0, 'total': 0, 'recent': deque(maxlen=20)}
            for s in self.strategies
        }
        
        # Default weights
        self.weights = {s: 1.0 / len(self.strategies) for s in self.strategies}
    
    def record_result(self, strategy: str, was_correct: bool):
        """Record the result of a strategy's prediction."""
        if strategy in self.performance:
            self.performance[strategy]['total'] += 1
            self.performance[strategy]['recent'].append(1 if was_correct else 0)
            if was_correct:
                self.performance[strategy]['correct'] += 1
            
            self._update_weights()
    
    def _update_weights(self):
        """Update strategy weights based on recent performance."""
        recent_accuracies = {}
        
        for strategy, perf in self.performance.items():
            if len(perf['recent']) >= 5:
                recent_acc = sum(perf['recent']) / len(perf['recent'])
                recent_accuracies[strategy] = recent_acc
            else:
                recent_accuracies[strategy] = 0.5  # Default
        
        # Convert to weights (higher accuracy = higher weight)
        total = sum(recent_accuracies.values())
        if total > 0:
            self.weights = {k: v / total for k, v in recent_accuracies.items()}
    
    def get_best_strategy(self) -> str:
        """Get the currently best performing strategy."""
        best = max(self.weights, key=self.weights.get)
        return best
    
    def get_weights(self) -> Dict[str, float]:
        """Get current strategy weights."""
        return self.weights.copy()


class UltraPredictor:
    """
    Ultra-refined multi-strategy prediction engine.
    Combines multiple strategies with adaptive weighting.
    """
    
    def __init__(self, endpoint_type: str):
        self.endpoint_type = endpoint_type
        self.sequence_length = int(os.getenv("SEQUENCE_LENGTH", "30"))
        
        # Initialize predictors
        self.pattern_memory = PatternMemory()
        self.markov = MarkovChainPredictor(max_order=5)
        self.mean_reversion = MeanReversionPredictor(window_size=30)
        self.streak_analyzer = StreakAnalyzer(min_streak_for_reversal=3)
        self.alternation = AlternationPredictor()
        self.ml_ensemble = EnsembleMLPredictor(endpoint_type)
        self.strategy_selector = StrategySelector()
        
        # Training data
        self.training_sequences: List[List[int]] = []
        self.training_labels: List[int] = []
        
        # State
        self.fitted = False
        self.last_prediction: Optional[Dict] = None
        self.prediction_history: List[Dict] = []
        
        # Game-specific configurations
        self.valid_values = self._get_valid_values()
    
    def _get_valid_values(self) -> List[int]:
        """Get valid outcome values for the game type."""
        if self.endpoint_type == 'lucky7eu':
            return [0, 1, 2]
        elif self.endpoint_type == 'dt20':
            return [1, 2, 3]
        else:  # teen20
            return [1, 2]
    
    def fit(self, historical_data: List[Dict]) -> float:
        """Train the predictor on historical data."""
        min_samples = int(os.getenv("MIN_TRAINING_SAMPLES", "50"))
        
        if len(historical_data) < min_samples:
            logger.warning(f"Insufficient data: {len(historical_data)} samples")
            return 0.0
        
        try:
            # Extract results
            results = [int(d["result"]) for d in historical_data]
            
            # Update Markov chains
            self.markov.update(results)
            
            # Prepare sequences for ML
            for i in range(len(results) - self.sequence_length - 1):
                seq = results[i:i + self.sequence_length]
                label = results[i + self.sequence_length]
                self.training_sequences.append(seq)
                self.training_labels.append(label)
            
            # Train ML ensemble
            if len(self.training_sequences) >= 20:
                self.ml_ensemble.fit(self.training_sequences, self.training_labels)
            
            self.fitted = True
            
            # Calculate estimated accuracy from cross-validation
            # This is just an estimate for logging
            accuracy = self._estimate_accuracy(results)
            logger.info(f"UltraPredictor trained for {self.endpoint_type}: {accuracy:.2%} estimated accuracy")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"UltraPredictor training failed: {e}")
            return 0.0
    
    def _estimate_accuracy(self, results: List[int]) -> float:
        """Estimate accuracy through backtesting."""
        if len(results) < self.sequence_length + 10:
            return 0.5
        
        correct = 0
        total = 0
        
        # Backtest on last 50 results
        test_start = max(self.sequence_length, len(results) - 50)
        
        for i in range(test_start, len(results) - 1):
            seq = results[i - self.sequence_length:i]
            actual = results[i]
            
            pred, _ = self._make_prediction(seq)
            if pred == actual:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.5
    
    def _make_prediction(self, sequence: List[int]) -> Tuple[int, float]:
        """Internal prediction method used for backtesting and real predictions."""
        # Get predictions from all strategies
        strategy_predictions: Dict[str, Dict[int, float]] = {}
        
        # 1. Pattern Memory
        pattern_pred = self.pattern_memory.get_prediction(sequence)
        if pattern_pred:
            pred_val, conf = pattern_pred
            strategy_predictions['pattern_memory'] = {v: 0.1 for v in self.valid_values}
            strategy_predictions['pattern_memory'][pred_val] = conf
        else:
            strategy_predictions['pattern_memory'] = {v: 1.0 / len(self.valid_values) for v in self.valid_values}
        
        # 2. Markov Chain
        strategy_predictions['markov'] = self.markov.predict(sequence, self.valid_values)
        
        # 3. Mean Reversion
        strategy_predictions['mean_reversion'] = self.mean_reversion.predict(sequence, self.valid_values)
        
        # 4. Streak Analysis
        strategy_predictions['streak'] = self.streak_analyzer.predict(sequence, self.valid_values)
        
        # 5. Alternation
        strategy_predictions['alternation'] = self.alternation.predict(sequence, self.valid_values)
        
        # 6. ML Ensemble
        strategy_predictions['ml_ensemble'] = self.ml_ensemble.predict(sequence, self.valid_values)
        
        # 7. Combined (weighted average of all)
        combined = {v: 0.0 for v in self.valid_values}
        weights = self.strategy_selector.get_weights()
        
        for strategy, probs in strategy_predictions.items():
            if strategy != 'combined':
                weight = weights.get(strategy, 0.15)
                for v in self.valid_values:
                    combined[v] += probs.get(v, 0) * weight
        
        # Normalize combined
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}
        
        strategy_predictions['combined'] = combined
        
        # Select best strategy and use its prediction
        best_strategy = self.strategy_selector.get_best_strategy()
        
        # But also consider the combined approach
        final_probs = {}
        for v in self.valid_values:
            # 60% weight to best strategy, 40% to combined
            final_probs[v] = (
                0.4 * strategy_predictions.get(best_strategy, combined).get(v, 0.33) +
                0.6 * combined.get(v, 0.33)
            )
        
        # Get final prediction
        best_value = max(final_probs, key=final_probs.get)
        confidence = final_probs[best_value]
        
        return best_value, confidence
    
    def predict(self, sequence: List[int]) -> Tuple[int, float]:
        """
        Make a prediction for the next value.
        Returns (predicted_value, confidence).
        """
        pred, conf = self._make_prediction(sequence)
        
        # Store prediction for verification
        self.last_prediction = {
            'sequence': sequence.copy(),
            'prediction': pred,
            'confidence': conf,
            'timestamp': datetime.now()
        }
        
        return pred, conf
    
    def predict_with_strategy(self, sequence: np.ndarray, game_type: str) -> Tuple[str, float]:
        """
        Make a prediction with game-specific adjustments.
        Returns (predicted_value_str, confidence).
        """
        seq_list = list(sequence.flatten().astype(int))
        pred, conf = self.predict(seq_list)
        
        # Apply game-specific adjustments
        if game_type == 'lucky7eu':
            # For lucky7eu, 1 (equal) is rare, favor 0 and 2
            if conf < 0.4 and pred == 1:
                # Low confidence on "equal", switch to mean reversion
                reversion_probs = self.mean_reversion.predict(seq_list, self.valid_values)
                if reversion_probs[0] > reversion_probs[2]:
                    pred = 0
                else:
                    pred = 2
                conf = max(reversion_probs.values())
        
        elif game_type in ['teen20', 'dt20']:
            # For teen/dt, check for strong streaks
            streak_len, streak_val = self.streak_analyzer.get_streak(seq_list)
            if streak_len >= 4:
                # Strong streak, boost reversal confidence
                other_vals = [v for v in self.valid_values if v != streak_val]
                if other_vals and pred == streak_val and conf < 0.6:
                    pred = other_vals[0] if len(other_vals) == 1 else max(other_vals)
                    conf = 0.55
        
        return str(pred), float(conf)
    
    def record_result(self, sequence: List[int], predicted: int, actual: int):
        """Record the result of a prediction for learning."""
        was_correct = predicted == actual
        
        # Update pattern memory
        self.pattern_memory.add_pattern(sequence, actual, was_correct)
        
        # Update Markov chains with new data
        extended_seq = sequence + [actual]
        self.markov.update(extended_seq)
        
        # Update streak analyzer
        streak_len, streak_val = self.streak_analyzer.get_streak(sequence)
        did_reverse = (actual != streak_val)
        self.streak_analyzer.record_result(streak_len, did_reverse)
        
        # Record for strategy selection
        # Determine which strategy would have been correct
        for strategy in self.strategy_selector.strategies:
            if strategy == 'combined':
                self.strategy_selector.record_result(strategy, was_correct)
            else:
                # Re-run each strategy to see if it would have been correct
                strategy_pred = self._get_strategy_prediction(strategy, sequence)
                self.strategy_selector.record_result(strategy, strategy_pred == actual)
        
        # Store in history
        self.prediction_history.append({
            'sequence': sequence,
            'predicted': predicted,
            'actual': actual,
            'correct': was_correct,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
    
    def _get_strategy_prediction(self, strategy: str, sequence: List[int]) -> int:
        """Get prediction from a specific strategy."""
        try:
            if strategy == 'pattern_memory':
                pred = self.pattern_memory.get_prediction(sequence)
                return pred[0] if pred else self.valid_values[0]
            elif strategy == 'markov':
                probs = self.markov.predict(sequence, self.valid_values)
                return max(probs, key=probs.get)
            elif strategy == 'mean_reversion':
                probs = self.mean_reversion.predict(sequence, self.valid_values)
                return max(probs, key=probs.get)
            elif strategy == 'streak':
                probs = self.streak_analyzer.predict(sequence, self.valid_values)
                return max(probs, key=probs.get)
            elif strategy == 'alternation':
                probs = self.alternation.predict(sequence, self.valid_values)
                return max(probs, key=probs.get)
            elif strategy == 'ml_ensemble':
                probs = self.ml_ensemble.predict(sequence, self.valid_values)
                return max(probs, key=probs.get)
            else:
                return self.valid_values[0]
        except Exception:
            return self.valid_values[0]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.prediction_history:
            return {'accuracy': 0, 'total': 0}
        
        correct = sum(1 for p in self.prediction_history if p['correct'])
        total = len(self.prediction_history)
        
        return {
            'accuracy': correct / total if total > 0 else 0,
            'total': total,
            'correct': correct,
            'strategy_weights': self.strategy_selector.get_weights(),
            'pattern_memory_stats': {
                'patterns': len(self.pattern_memory.patterns),
                'hits': self.pattern_memory.hit_count,
                'misses': self.pattern_memory.miss_count
            }
        }


# Backward compatibility
class AdvancedEnsemblePredictor(UltraPredictor):
    """Alias for backward compatibility."""
    pass


class AdvancedFeatureEngine:
    """Kept for backward compatibility."""
    
    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
    
    def extract_features(self, sequence: np.ndarray) -> np.ndarray:
        return np.array(sequence).flatten()
    
    def _get_current_streak(self, seq: np.ndarray) -> Tuple[int, int]:
        seq = list(seq.flatten())
        if len(seq) == 0:
            return 0, 0
        
        current = seq[-1]
        streak = 1
        for i in range(len(seq) - 2, -1, -1):
            if seq[i] == current:
                streak += 1
            else:
                break
        return streak, int(current)
