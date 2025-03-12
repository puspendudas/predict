import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import requests
import json
import os
import joblib
from pathlib import Path
from .config.database import Database
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd

class PredictionService:
    def __init__(self):
        self.model_path = Path("./ml_model/prediction_model.pkl")
        self.scaler_path = Path("./ml_model/scaler.pkl")
        self.metrics_path = Path("./ml_model/metrics_history.json")
        
        # Initialize database
        self.db = Database()
        self.last_predictions = {}
        
        # Model parameters
        self.min_confidence_threshold = 0.65  # Increased from 0.6
        self.min_training_samples = 50  # Increased from 30
        self.sequence_length = 10  # Increased from 8
        self.update_threshold = 0.70  # Minimum accuracy required for model update
        
        # Load or initialize models
        self._load_or_initialize_models()
        
        # Training schedule
        self.last_training_time = datetime.now()
        self.training_interval = timedelta(hours=1)  # Train every hour if needed
        self.last_accuracy_check = datetime.now()
        self.accuracy_check_interval = timedelta(minutes=15)  # Check accuracy every 15 minutes

    def _load_or_initialize_models(self):
        """Load existing models or initialize new ones."""
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                self.ensemble = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logging.info("Loaded existing model and scaler from files")
                return
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
        
        self._initialize_new_models()

    def _initialize_new_models(self):
        """Initialize new models with optimized parameters."""
        self.rf_model = RandomForestClassifier(
            n_estimators=300,  # Increased from 200
            max_depth=15,      # Increased from 12
            min_samples_split=5,
            class_weight='balanced',  # Added class weight
            random_state=42
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=300,  # Increased from 200
            learning_rate=0.08,  # Decreased from 0.1 for better generalization
            max_depth=10,      # Increased from 8
            subsample=0.8,     # Added subsample
            random_state=42
        )
        
        self.ada_model = AdaBoostClassifier(  # Added AdaBoost
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )
        
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('gb', self.gb_model),
                ('ada', self.ada_model)
            ],
            voting='soft',
            weights=[2, 1, 1]  # Give more weight to RandomForest
        )
        
        self.scaler = StandardScaler()
        logging.info("Initialized new optimized models")

    def save_models(self):
        """Save the trained models to disk."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.ensemble, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logging.info("Models saved successfully")
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")

    def extract_features(self, sequence: List[int], sequence_length: int = 10) -> np.ndarray:
        """Extract advanced features from the sequence."""
        if len(sequence) < sequence_length:
            return np.zeros((1, sequence_length * 4))  # Increased feature space
        
        features = []
        # Basic sequence
        features.extend(sequence[-sequence_length:])
        
        # Rolling statistics with multiple windows
        for window in [3, 5, 7]:
            series = pd.Series(sequence)
            rolling_mean = series.rolling(window).mean().fillna(0).values[-sequence_length:]
            rolling_std = series.rolling(window).std().fillna(0).values[-sequence_length:]
            features.extend(rolling_mean)
            features.extend(rolling_std)
        
        # Add pattern features
        pattern_features = self._extract_pattern_features(sequence)
        features.extend(pattern_features)
        
        return np.array(features).reshape(1, -1)

    def _extract_pattern_features(self, sequence: List[int]) -> List[float]:
        """Extract pattern-based features."""
        pattern_features = []
        
        # Frequency of each outcome in last n positions
        for n in [5, 10, 15]:
            last_n = sequence[-n:] if len(sequence) >= n else sequence
            for i in range(3):  # For 0, 1, 2 outcomes
                freq = last_n.count(i) / len(last_n)
                pattern_features.append(freq)
        
        # Streak features
        current_streak = 1
        max_streak = 1
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        pattern_features.extend([current_streak / len(sequence), max_streak / len(sequence)])
        
        return pattern_features

    def prepare_data(self, data: List[Dict], sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data with advanced feature engineering."""
        if len(data) < sequence_length + 1:
            return np.array([]), np.array([])
        
        X, y = [], []
        results = [int(d["result"]) for d in data]
        
        for i in range(len(results) - sequence_length):
            sequence = results[i:i+sequence_length]
            features = self.extract_features(sequence, sequence_length)
            X.append(features.flatten())
            y.append(results[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) > 0:
            X = self.scaler.fit_transform(X)
        
        return X, y

    def check_and_update_accuracy(self) -> Dict:
        """Check accuracy and trigger model update if needed."""
        current_time = datetime.now()
        
        # Check if it's time to verify accuracy
        if (current_time - self.last_accuracy_check) < self.accuracy_check_interval:
            return {}
            
        try:
            metrics = self.db.get_accuracy_metrics(last_n_days=1)  # Check last 24 hours
            accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
            
            # Save accuracy metrics
            self.db.save_accuracy_metrics(accuracy, metrics["total"])
            
            # Update model if accuracy is below threshold and enough time has passed
            if (accuracy < self.update_threshold and 
                (current_time - self.last_training_time) >= self.training_interval):
                self.update_model_if_needed(force_update=True)
            
            self.last_accuracy_check = current_time
            
            return {
                "accuracy": accuracy,
                "total_predictions": metrics["total"],
                "correct_predictions": metrics["correct"],
                "last_check": current_time.isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error checking accuracy: {str(e)}")
            return {}

    def update_model_if_needed(self, force_update: bool = False) -> None:
        """Update model with improved training process."""
        try:
            if not force_update:
                metrics = self.db.get_accuracy_metrics(last_n_days=1)
                should_update = (metrics["total"] >= 30 and 
                               (metrics["correct"] / metrics["total"]) < self.update_threshold)
                
                if not should_update:
                    return
            
            historical_data = self.db.get_last_n_results(500)  # Increased historical data
            if len(historical_data) < self.min_training_samples:
                return
            
            X, y = self.prepare_data(historical_data, sequence_length=self.sequence_length)
            if len(X) < self.min_training_samples:
                return
            
            # Split data with stratification
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train ensemble
            self.ensemble.fit(X_train, y_train)
            
            # Validate performance
            val_pred = self.ensemble.predict(X_val)
            val_score = accuracy_score(y_val, val_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, val_pred, average='weighted'
            )
            
            logging.info(f"Validation metrics - Accuracy: {val_score:.3f}, "
                        f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            if val_score < 0.5:
                logging.warning("Model validation score too low, keeping previous model")
                return
            
            # Save the updated models
            self.save_models()
            self.last_training_time = datetime.now()
            
            # Save training metrics
            self._save_training_metrics({
                "timestamp": datetime.now().isoformat(),
                "accuracy": val_score,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "samples": len(X)
            })
            
            logging.info(f"Model retrained with {len(X)} samples")
            
        except Exception as e:
            logging.error(f"Error updating model: {str(e)}")

    def _save_training_metrics(self, metrics: Dict) -> None:
        """Save training metrics history."""
        try:
            history = []
            if self.metrics_path.exists():
                with open(self.metrics_path, 'r') as f:
                    history = json.load(f)
            
            history.append(metrics)
            # Keep only last 100 training sessions
            history = history[-100:]
            
            with open(self.metrics_path, 'w') as f:
                json.dump(history, f)
                
        except Exception as e:
            logging.error(f"Error saving training metrics: {str(e)}")

    def predict_next_rounds(self, n_predictions: int = 2) -> List[Dict]:
        """Make predictions with confidence scoring and fallback strategies."""
        try:
            historical_data = self.db.get_last_n_results(150)
            if len(historical_data) < self.min_training_samples:
                default_predictions = [{"prediction": "0", "confidence": 0.0} for _ in range(n_predictions)]
                self._save_predictions_to_db(default_predictions)
                return default_predictions
            
            current_time = datetime.now().isoformat()
            last_sequence = [int(d["result"]) for d in historical_data[-10:]]
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for i in range(n_predictions):
                features = self.extract_features(current_sequence)
                features_scaled = self.scaler.transform(features)
                
                # Get probabilities from ensemble model
                pred_proba = self.ensemble.predict_proba(features_scaled)[0]
                confidence = np.max(pred_proba)
                pred = str(np.argmax(pred_proba))
                
                prediction_data = {
                    "prediction": pred,
                    "confidence": float(confidence),
                    "timestamp": current_time,
                    "model_version": self._get_model_version(),
                    "features_used": len(features.flatten()),
                    "probabilities": {str(i): float(p) for i, p in enumerate(pred_proba)}
                }
                
                # If confidence is low, use alternative strategy
                if confidence < self.min_confidence_threshold:
                    recent_results = [int(d["result"]) for d in historical_data[-20:]]
                    pred = str(max(set(recent_results), key=recent_results.count))
                    prediction_data.update({
                        "prediction": pred,
                        "fallback_used": True,
                        "original_prediction": prediction_data["prediction"],
                        "fallback_reason": "low_confidence"
                    })
                    logging.warning(f"Low confidence ({confidence:.2f}), using fallback prediction: {pred}")
                
                # Add sequence information
                prediction_data.update({
                    "sequence_used": current_sequence.copy(),
                    "mid": str(int(historical_data[0]["mid"]) + i + 1)
                })
                
                predictions.append(prediction_data)
                current_sequence = current_sequence[1:] + [int(pred)]
            
            # Save predictions to database
            self._save_predictions_to_db(predictions)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            default_predictions = [{"prediction": "0", "confidence": 0.0} for _ in range(n_predictions)]
            self._save_predictions_to_db(default_predictions)
            return default_predictions

    def _save_predictions_to_db(self, predictions: List[Dict]) -> None:
        """Save detailed prediction information to database."""
        try:
            current_time = datetime.now().isoformat()
            
            for pred_data in predictions:
                # Ensure all required fields are present
                prediction_record = {
                    "mid": pred_data.get("mid", str(int(datetime.now().timestamp() * 1000))),
                    "prediction": pred_data["prediction"],
                    "confidence": pred_data.get("confidence", 0.0),
                    "timestamp": pred_data.get("timestamp", current_time),
                    "model_version": pred_data.get("model_version", self._get_model_version()),
                    "features_used": pred_data.get("features_used", 0),
                    "probabilities": pred_data.get("probabilities", {}),
                    "sequence_used": pred_data.get("sequence_used", []),
                    "verified": False,
                    "actual_result": None,
                    "was_correct": None
                }
                
                # Add fallback information if present
                if "fallback_used" in pred_data:
                    prediction_record.update({
                        "fallback_used": True,
                        "original_prediction": pred_data["original_prediction"],
                        "fallback_reason": pred_data["fallback_reason"]
                    })
                
                # Save to database
                self.db.save_prediction_details(prediction_record)
                
                # Update last predictions cache
                self.last_predictions[prediction_record["mid"]] = prediction_record["prediction"]
                
        except Exception as e:
            logging.error(f"Error saving predictions to database: {str(e)}")

    def _get_model_version(self) -> str:
        """Get current model version based on training timestamp."""
        try:
            model_stats = os.stat(self.model_path)
            return datetime.fromtimestamp(model_stats.st_mtime).isoformat()
        except Exception:
            return datetime.now().isoformat()

    def verify_prediction(self, mid: str, actual_result: str) -> None:
        """Verify prediction and update accuracy metrics."""
        try:
            # Update prediction result in database
            was_correct = self.db.update_prediction_result(mid, actual_result)
            
            if was_correct is not None:
                # Get prediction details
                prediction_details = self.db.get_prediction_details(mid)
                
                if prediction_details:
                    # Update accuracy metrics with detailed information
                    self.db.update_accuracy_metrics({
                        "timestamp": datetime.now().isoformat(),
                        "mid": mid,
                        "was_correct": was_correct,
                        "confidence": prediction_details.get("confidence", 0.0),
                        "fallback_used": prediction_details.get("fallback_used", False),
                        "model_version": prediction_details.get("model_version", "unknown")
                    })
                
                # Check if model update is needed
                self.update_model_if_needed()
                
        except Exception as e:
            logging.error(f"Error verifying prediction: {str(e)}")

    def fetch_latest_data(self):
        try:
            response = requests.get(os.getenv("API_URL"))
            response.raise_for_status()  # Raise exception for non-200 status codes
            data = response.json()
            
            # Validate response structure
            if not data.get("data", {}).get("data", {}).get("data", {}).get("result"):
                raise ValueError("Invalid response format")
                
            results = data["data"]["data"]["data"]["result"]
            
            # Verify previous predictions and save results
            for result in results:
                if "mid" not in result or "result" not in result:
                    continue
                self.verify_prediction(result["mid"], result["result"])
                self.db.insert_result(result)
            
            return results
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            return []
