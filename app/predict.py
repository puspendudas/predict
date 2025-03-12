import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import json
import os
from .config.database import Database
from datetime import datetime
import logging

class PredictionService:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.db = Database()
        self.last_predictions = {}

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

    def verify_prediction(self, mid: str, actual_result: str):
        was_correct = self.db.update_prediction_result(mid, actual_result)
        if was_correct is not None:
            self.update_model_if_needed()

    def update_model_if_needed(self, force_update=False):
        metrics = self.db.get_accuracy_metrics(last_n_days=1)
        should_update = (
            force_update or 
            (metrics["total"] >= 10 and (metrics["correct"] / metrics["total"]) < 0.6)
        )
        
        if should_update:
            try:
                # Get more historical data for better training
                historical_data = self.db.get_last_n_results(200)  # Increased from 100
                if len(historical_data) < 20:  # Minimum required data
                    return
                
                X, y = self.prepare_data(historical_data, sequence_length=8)  # Increased sequence length
                if len(X) > 0:
                    self.model = RandomForestClassifier(
                        n_estimators=300,  # Increased from 200
                        max_depth=10,
                        min_samples_split=5,
                        random_state=42
                    )
                    self.model.fit(X, y)
                    logging.info(f"Model retrained with {len(X)} samples")
            except Exception as e:
                logging.error(f"Error updating model: {str(e)}")

    def prepare_data(self, data, sequence_length=5):
        X, y = [], []
        results = [int(d["result"]) for d in data]
        
        for i in range(len(results) - sequence_length):
            X.append(results[i:i+sequence_length])
            y.append(results[i+sequence_length])
            
        return np.array(X), np.array(y)

    def predict_next_rounds(self, n_predictions=2):
        historical_data = self.db.get_last_n_results(100)  # Increased from 50
        if len(historical_data) < 10:
            return ["0"] * n_predictions
            
        X, y = self.prepare_data(historical_data, sequence_length=8)  # Match sequence length
        current_time = datetime.now().isoformat()
        
        if len(X) > 0:
            # Ensure model is fitted
            if not hasattr(self.model, "fitted_") or not self.model.fitted_:
                self.model.fit(X, y)
            
            last_sequence = np.array([int(d["result"]) for d in historical_data[-8:]])  # Match sequence length
            
            predictions = []
            probabilities = []
            
            for _ in range(n_predictions):
                pred_proba = self.model.predict_proba([last_sequence])[0]
                pred = str(np.argmax(pred_proba))
                confidence = np.max(pred_proba)
                
                if confidence < 0.4:  # Low confidence threshold
                    logging.warning(f"Low confidence prediction: {confidence}")
                
                predictions.append(pred)
                last_sequence = np.append(last_sequence[1:], int(pred))
            
            # Save predictions
            next_mid = str(int(historical_data[0]["mid"]) + 1)
            for i, pred in enumerate(predictions):
                pred_mid = str(int(next_mid) + i)
                self.db.save_prediction(pred_mid, pred, current_time)
                self.last_predictions[pred_mid] = pred
            
            return predictions
        
        return ["0"] * n_predictions

    def check_and_update_accuracy(self):
        try:
            # Get accuracy metrics for all time
            metrics = self.db.get_accuracy_metrics(last_n_days=None)
            accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
            
            # Save accuracy metrics
            self.db.save_accuracy_metrics(accuracy, metrics["total"])
            
            # Update model if accuracy is below threshold
            if accuracy < 0.6 and metrics["total"] >= 50:
                self.update_model_if_needed(force_update=True)
                
            return {
                "accuracy": accuracy,
                "total_predictions": metrics["total"],
                "correct_predictions": metrics["correct"]
            }
        except Exception as e:
            logging.error(f"Error checking accuracy: {str(e)}")
            return {"accuracy": 0, "total_predictions": 0, "correct_predictions": 0}
