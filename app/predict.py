import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import json
import os
from .config.database import Database
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import time

class PredictionService:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.db = Database()
        self.last_predictions = {}
        self.endpoints = {
            'teen20': os.getenv("TEEN20_API_URL"),
            'lucky7eu': os.getenv("LUCKY7EU_API_URL"),
            'dt20': os.getenv("DT20_API_URL")
        }
        self.verification_interval = 5  # seconds
        self.min_confidence_threshold = 0.4
        self.accuracy_threshold = 0.6
        self.min_samples_for_training = 20
        self.max_samples_for_training = 500
        self.sequence_length = 8
        self.prediction_window = 2

    def fetch_latest_data(self, endpoint_type='teen20') -> List[Dict]:
        try:
            api_url = self.endpoints.get(endpoint_type)
            if not api_url:
                raise ValueError(f"Invalid endpoint type: {endpoint_type}")
                
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            
            # Handle different response formats
            if endpoint_type == 'lucky7eu':
                if not data.get("data", {}).get("data", {}).get("data", {}).get("result"):
                    raise ValueError("Invalid Lucky7EU response format")
                results = data["data"]["data"]["data"]["result"]
            elif endpoint_type == 'dt20':
                if not data.get("data", {}).get("data", {}).get("data", {}).get("result"):
                    raise ValueError("Invalid DT20 response format")
                results = data["data"]["data"]["data"]["result"]
            else:
                if not data.get("data", {}).get("data", {}).get("data", {}).get("result"):
                    raise ValueError("Invalid Teen20 response format")
                results = data["data"]["data"]["data"]["result"]
            
            return results
        except Exception as e:
            logging.error(f"Error fetching data from {endpoint_type}: {str(e)}")
            return []

    def verify_predictions(self, endpoint_type: str) -> None:
        """Continuously verify predictions against actual results."""
        while True:
            try:
                # Fetch latest results
                results = self.fetch_latest_data(endpoint_type)
                if not results:
                    time.sleep(self.verification_interval)
                    continue

                # Verify each result against our predictions
                for result in results:
                    if "mid" not in result or "result" not in result:
                        continue
                    
                    # Check if we have a prediction for this result
                    prediction = self.db.get_prediction(result["mid"], endpoint_type)
                    if prediction:
                        was_correct = prediction["predicted_value"] == result["result"]
                        
                        # Update prediction result in database
                        self.db.update_prediction_result(
                            result["mid"],
                            result["result"],
                            endpoint_type,
                            was_correct
                        )
                        
                        # Log the verification result
                        logging.info(
                            f"Prediction verification for {endpoint_type} - MID: {result['mid']}, "
                            f"Predicted: {prediction['predicted_value']}, "
                            f"Actual: {result['result']}, "
                            f"Correct: {was_correct}"
                        )
                        
                        # Check if we need to retrain the model
                        self.check_and_update_model(endpoint_type)
                    
                    # Save the actual result
                    self.db.insert_result(result, endpoint_type)
                
                time.sleep(self.verification_interval)
            except Exception as e:
                logging.error(f"Error in verification loop for {endpoint_type}: {str(e)}")
                time.sleep(self.verification_interval)

    def check_and_update_model(self, endpoint_type: str) -> None:
        """Check model performance and update if necessary."""
        try:
            # Get recent accuracy metrics
            metrics = self.db.get_accuracy_metrics(endpoint_type, last_n_days=1)
            if metrics["total"] < 10:
                return

            accuracy = metrics["correct"] / metrics["total"]
            
            # Check if model needs updating
            should_update = (
                accuracy < self.accuracy_threshold or
                self.db.get_consecutive_incorrect_predictions(endpoint_type) >= 5
            )
            
            if should_update:
                self.update_model(endpoint_type)
                
        except Exception as e:
            logging.error(f"Error checking model for {endpoint_type}: {str(e)}")

    def update_model(self, endpoint_type: str) -> None:
        """Update the model with recent data."""
        try:
            # Get historical data for training
            historical_data = self.db.get_last_n_results(
                self.max_samples_for_training,
                endpoint_type
            )
            
            if len(historical_data) < self.min_samples_for_training:
                logging.warning(f"Insufficient data for model update: {len(historical_data)} samples")
                return
            
            # Prepare training data
            X, y = self.prepare_data(historical_data, self.sequence_length)
            if len(X) == 0:
                return
            
            # Train new model with improved parameters
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            )
            
            self.model.fit(X, y)
            logging.info(
                f"Model updated for {endpoint_type} with {len(X)} samples. "
                f"Current accuracy: {self.calculate_accuracy(X, y):.2f}"
            )
            
        except Exception as e:
            logging.error(f"Error updating model for {endpoint_type}: {str(e)}")

    def calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model accuracy on training data."""
        predictions = self.model.predict(X)
        return np.mean(predictions == y)

    def prepare_data(self, data: List[Dict], sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical results."""
        X, y = [], []
        results = [int(d["result"]) for d in data]
        
        for i in range(len(results) - sequence_length):
            X.append(results[i:i+sequence_length])
            y.append(results[i+sequence_length])
            
        return np.array(X), np.array(y)

    def predict_next_rounds(self, endpoint_type='teen20', n_predictions=2) -> List[str]:
        """Generate predictions for the next rounds."""
        historical_data = self.db.get_last_n_results(100, endpoint_type)
        if len(historical_data) < 10:
            return ["0"] * n_predictions
            
        X, y = self.prepare_data(historical_data, self.sequence_length)
        current_time = datetime.now().isoformat()
        
        if len(X) > 0:
            # Ensure model is fitted
            if not hasattr(self.model, "fitted_") or not self.model.fitted_:
                self.update_model(endpoint_type)
            
            last_sequence = np.array([int(d["result"]) for d in historical_data[-self.sequence_length:]])
            
            predictions = []
            for _ in range(n_predictions):
                pred_proba = self.model.predict_proba([last_sequence])[0]
                pred = str(np.argmax(pred_proba))
                confidence = np.max(pred_proba)
                
                if confidence < self.min_confidence_threshold:
                    logging.warning(
                        f"Low confidence prediction for {endpoint_type}: {confidence}. "
                        f"Consider retraining model."
                    )
                
                predictions.append(pred)
                last_sequence = np.append(last_sequence[1:], int(pred))
            
            # Save predictions
            next_mid = str(int(historical_data[0]["mid"]) + 1)
            for i, pred in enumerate(predictions):
                pred_mid = str(int(next_mid) + i)
                self.db.save_prediction(pred_mid, pred, current_time, endpoint_type)
                self.last_predictions[pred_mid] = pred
            
            return predictions
        
        return ["0"] * n_predictions

    def start_verification_loop(self, endpoint_type: str) -> None:
        """Start the continuous verification process."""
        import threading
        thread = threading.Thread(
            target=self.verify_predictions,
            args=(endpoint_type,),
            daemon=True
        )
        thread.start()
