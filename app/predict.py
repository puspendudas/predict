import numpy as np
from sklearn.ensemble import RandomForestClassifier
import aiohttp
import asyncio
import json
import os
from .config.database import Database
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import time
import urllib3
import certifi
from functools import lru_cache
from app.config.logging_config import setup_logging

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize logging
logger = setup_logging()

class PredictionService:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.db = Database()
        self.last_predictions = {}
        self.last_mids = {}
        self.prediction_cache = {}
        self.cache_timeout = 30  # seconds
        self.endpoints = {
            'teen20': os.getenv('TEEN20_ODDS_API_URL'),
            'lucky7eu': os.getenv('LUCKY7EU_ODDS_API_URL'),
            'dt20': os.getenv('DT20_ODDS_API_URL')
        }
        self.result_endpoints = {
            'teen20': os.getenv('TEEN20_RESULTS_API_URL'),
            'lucky7eu': os.getenv('LUCKY7EU_RESULTS_API_URL'),
            'dt20': os.getenv('DT20_RESULTS_API_URL')
        }
        self.verification_interval = 5
        self.prediction_interval = 30
        self.min_confidence_threshold = 0.4
        self.accuracy_threshold = 0.6
        self.min_samples_for_training = 20
        self.max_samples_for_training = 500
        self.sequence_length = 8
        self.prediction_window = 2
        self.session = None
        self.lock = asyncio.Lock()

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def fetch_latest_data(self, endpoint_type='teen20') -> Tuple[List[Dict], Optional[str], Optional[Dict]]:
        """Fetch latest data asynchronously and return results, current MID, and game info."""
        try:
            await self.init_session()
            odds_url = self.endpoints.get(endpoint_type)
            if not odds_url:
                raise ValueError(f"Invalid endpoint type: {endpoint_type}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json',
                'Origin': 'https://terminal.apiserver.digital',
                'Referer': 'https://terminal.apiserver.digital/',
            }
            
            async with self.session.get(odds_url, headers=headers, ssl=False) as response:
                odds_data = await response.json()
            
            current_mid = None
            game_info = None
            t1_data = odds_data.get("data", {}).get("data", {}).get("data", {}).get("t1", [])
            
            if t1_data and len(t1_data) > 0:
                t1_data = t1_data[0]
                current_mid = t1_data.get("mid")
                game_info = {
                    "mid": current_mid,
                    "autotime": t1_data.get("autotime"),
                    "gtype": t1_data.get("gtype"),
                    "max": t1_data.get("max"),
                    "min": t1_data.get("min")
                }
            
            results_url = self.result_endpoints.get(endpoint_type)
            if not results_url:
                raise ValueError(f"Invalid results endpoint type: {endpoint_type}")
            
            async with self.session.get(results_url, headers=headers, ssl=False) as response:
                results_data = await response.json()
            
            results = results_data.get("data", {}).get("data", {}).get("data", [])
            return results, current_mid, game_info
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return [], None, None

    @lru_cache(maxsize=100)
    def get_cached_predictions(self, mid: str, endpoint_type: str) -> Optional[List[str]]:
        """Get cached predictions if they exist and are not expired."""
        if mid in self.prediction_cache:
            cache_entry = self.prediction_cache[mid]
            if time.time() - cache_entry['timestamp'] < self.cache_timeout:
                return cache_entry['predictions']
        return None

    def cache_predictions(self, mid: str, predictions: List[str]):
        """Cache predictions with timestamp."""
        self.prediction_cache[mid] = {
            'predictions': predictions,
            'timestamp': time.time()
        }

    async def verify_predictions(self, endpoint_type: str) -> None:
        """Asynchronously verify predictions."""
        try:
            results, current_mid, _ = await self.fetch_latest_data(endpoint_type)
            if not results or not current_mid:
                return

            for result in results:
                result_mid = result["mid"]
                actual_value = result["result"]
                
                prediction = self.db.prediction_history.find_one({
                    "mid": result_mid,
                    "endpoint_type": endpoint_type,
                    "verified": False
                })
                
                if prediction:
                    predicted_value = prediction["predicted_value"]
                    was_correct = actual_value == predicted_value
                    
                    await self.db.update_prediction_result(
                        result_mid,
                        actual_value,
                        endpoint_type,
                        was_correct
                    )
                    
                    self.db.insert_result(result, endpoint_type)
        except Exception as e:
            logger.error(f"Error in verification: {str(e)}")

    @lru_cache(maxsize=1000)
    def prepare_data(self, data: List[Dict], sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training with caching."""
        if not data:
            return np.array([]), np.array([])
            
        X = []
        y = []
        
        for i in range(len(data) - sequence_length):
            sequence = data[i:i + sequence_length]
            target = data[i + sequence_length]
            
            features = [float(item.get("result", 0)) for item in sequence]
            target_value = float(target.get("result", 0))
            
            X.append(features)
            y.append(target_value)
            
        return np.array(X), np.array(y)

    async def generate_prediction_for_mid(self, mid: str, endpoint_type: str) -> None:
        """Generate prediction for a specific MID asynchronously."""
        try:
            # Check cache first
            cached_prediction = self.get_cached_predictions(mid, endpoint_type)
            if cached_prediction:
                return

            results, current_mid, game_info = await self.fetch_latest_data(endpoint_type)
            if not results or not game_info:
                return

            # Prepare data for prediction
            X, y = self.prepare_data(results, self.sequence_length)
            if len(X) < self.min_samples_for_training:
                return

            # Train model if needed
            if len(X) > self.min_samples_for_training:
                self.model.fit(X, y)

            # Generate prediction
            last_sequence = results[-self.sequence_length:]
            features = [float(item.get("result", 0)) for item in last_sequence]
            prediction = self.model.predict([features])[0]
            
            # Cache the prediction
            self.cache_predictions(mid, [str(prediction)])
            
            # Store prediction in database
            self.db.insert_prediction({
                "mid": mid,
                "predicted_value": str(prediction),
                "endpoint_type": endpoint_type,
                "timestamp": datetime.now(),
                "verified": False
            })

        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")

    async def start_prediction_loop(self, endpoint_type: str):
        """Start the prediction loop asynchronously."""
        while True:
            try:
                results, current_mid, _ = await self.fetch_latest_data(endpoint_type)
                if current_mid and current_mid != self.last_mids.get(endpoint_type):
                    await self.generate_prediction_for_mid(current_mid, endpoint_type)
                    self.last_mids[endpoint_type] = current_mid
                await asyncio.sleep(self.prediction_interval)
            except Exception as e:
                logger.error(f"Error in prediction loop: {str(e)}")
                await asyncio.sleep(self.prediction_interval)

    async def start_verification_loop(self, endpoint_type: str):
        """Start the verification loop asynchronously."""
        while True:
            try:
                await self.verify_predictions(endpoint_type)
                await asyncio.sleep(self.verification_interval)
            except Exception as e:
                logger.error(f"Error in verification loop: {str(e)}")
                await asyncio.sleep(self.verification_interval)

    async def start_all_loops(self):
        """Start all prediction and verification loops."""
        tasks = []
        for endpoint_type in self.endpoints.keys():
            tasks.append(self.start_prediction_loop(endpoint_type))
            tasks.append(self.start_verification_loop(endpoint_type))
        await asyncio.gather(*tasks)

    def generate_prediction_for_mid(self, mid: str, endpoint_type: str) -> None:
        """Generate prediction for a specific MID."""
        try:
            # Get historical data for prediction
            historical_data = self.db.get_last_n_results(10000, endpoint_type)
            if len(historical_data) < self.min_samples_for_training:
                logging.warning(f"Insufficient historical data for {endpoint_type}: {len(historical_data)} samples")
                return
            
            # Ensure model is fitted
            if not hasattr(self.model, "fitted_") or not self.model.fitted_:
                logging.info(f"Training model for {endpoint_type} with {len(historical_data)} samples")
                self.update_model(endpoint_type)
            
            # Prepare data and generate prediction
            X, y = self.prepare_data(historical_data, self.sequence_length)
            if len(X) == 0:
                logging.warning(f"No valid sequences found for {endpoint_type}")
                return
            
            last_sequence = np.array([int(d["result"]) for d in historical_data[-self.sequence_length:]])
            pred_proba = self.model.predict_proba([last_sequence])[0]
            
            # Handle predictions based on game type
            if endpoint_type in ['teen20', 'dt20']:
                # For teen20 and dt20, only predict 1 or 2
                prob_1 = pred_proba[1] if len(pred_proba) > 1 else 0
                prob_2 = pred_proba[2] if len(pred_proba) > 2 else 0
                
                # Choose between 1 and 2 based on higher probability
                if prob_1 >= prob_2:
                    pred = "1"
                    confidence = float(prob_1)
                else:
                    pred = "2"
                    confidence = float(prob_2)
            else:  # lucky7eu
                # For lucky7eu, predict 0, 1, or 2
                prob_0 = pred_proba[0] if len(pred_proba) > 0 else 0
                prob_1 = pred_proba[1] if len(pred_proba) > 1 else 0
                prob_2 = pred_proba[2] if len(pred_proba) > 2 else 0
                
                # Choose between 0, 1, and 2 based on highest probability
                max_prob = max(prob_0, prob_1, prob_2)
                if max_prob == prob_0:
                    pred = "0"
                    confidence = float(prob_0)
                elif max_prob == prob_1:
                    pred = "1"
                    confidence = float(prob_1)
                else:
                    pred = "2"
                    confidence = float(prob_2)
            
            if confidence < self.min_confidence_threshold:
                logging.warning(
                    f"Low confidence prediction for {endpoint_type}: {confidence:.2f}. "
                    f"Consider retraining model."
                )
            
            # Save prediction
            current_time = datetime.now().isoformat()
            save_result = self.db.save_prediction(mid, pred, current_time, endpoint_type, confidence)
            
            if save_result:
                self.last_predictions[mid] = pred
                logging.info(
                    f"Successfully saved prediction for {endpoint_type} - "
                    f"MID: {mid}, Value: {pred}, Confidence: {confidence}"
                )
            else:
                logging.error(f"Failed to save prediction for {endpoint_type} - MID: {mid}")
            
        except Exception as e:
            logging.error(f"Error generating prediction for {endpoint_type} - MID: {mid}: {str(e)}")

    def check_and_update_model(self, endpoint_type: str) -> None:
        """Check model performance and update if necessary."""
        try:
            # Get recent accuracy metrics
            metrics = self.db.get_accuracy_metrics(endpoint_type, last_n_days=1)
            if metrics["total"] < 10:
                return

            accuracy = metrics["correct"] / metrics["total"]
            
            # Get recent accuracy trend
            trend = self.db.get_recent_accuracy_trend(endpoint_type, days=7)
            
            # Check if model needs updating based on multiple factors
            should_update = (
                accuracy < self.accuracy_threshold or
                self.db.get_consecutive_incorrect_predictions(endpoint_type) >= 5 or
                (trend["avg_accuracy"] < self.accuracy_threshold and trend["samples"] >= 5) or
                (trend["min_accuracy"] < 0.3 and trend["samples"] >= 10)
            )
            
            if should_update:
                logging.info(
                    f"Model update triggered for {endpoint_type}. "
                    f"Current accuracy: {accuracy:.2f}, "
                    f"Average accuracy: {trend['avg_accuracy']:.2f}, "
                    f"Min accuracy: {trend['min_accuracy']:.2f}"
                )
                self.update_model(endpoint_type)
                
                # Save the new accuracy metrics
                self.db.save_accuracy_metrics(
                    accuracy=accuracy,
                    total_predictions=metrics["total"],
                    endpoint_type=endpoint_type
                )
                
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
            
            # Train the model
            self.model.fit(X, y)
            
            # Calculate and log the accuracy
            accuracy = self.calculate_accuracy(X, y)
            logging.info(
                f"Model updated for {endpoint_type} with {len(X)} samples. "
                f"Current accuracy: {accuracy:.2f}"
            )
            
            # Save the accuracy metrics
            self.db.save_accuracy_metrics(
                accuracy=accuracy,
                total_predictions=len(X),
                endpoint_type=endpoint_type
            )
            
        except Exception as e:
            logging.error(f"Error updating model for {endpoint_type}: {str(e)}")

    def calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model accuracy on training data."""
        predictions = self.model.predict(X)
        return np.mean(predictions == y)

    def predict_next_rounds(self, endpoint_type='teen20', n_predictions=2) -> List[str]:
        """Generate predictions for the next rounds."""
        try:
            historical_data = self.db.get_last_n_results(10000, endpoint_type)
            if len(historical_data) < self.min_samples_for_training:
                logging.warning(f"Insufficient historical data for {endpoint_type}: {len(historical_data)} samples")
                return ["0"] * n_predictions
                
            # Always ensure model is fitted before making predictions
            if not hasattr(self.model, "fitted_") or not self.model.fitted_:
                logging.info(f"Training model for {endpoint_type} with {len(historical_data)} samples")
                self.update_model(endpoint_type)
            
            X, y = self.prepare_data(historical_data, self.sequence_length)
            if len(X) == 0:
                logging.warning(f"No valid sequences found for {endpoint_type}")
                return ["0"] * n_predictions
            
            current_time = datetime.now().isoformat()
            last_sequence = np.array([int(d["result"]) for d in historical_data[-self.sequence_length:]])
            
            predictions = []
            for _ in range(n_predictions):
                pred_proba = self.model.predict_proba([last_sequence])[0]
                
                # Handle predictions based on game type
                if endpoint_type in ['teen20', 'dt20']:
                    # For teen20 and dt20, only predict 1 or 2
                    prob_1 = pred_proba[1] if len(pred_proba) > 1 else 0
                    prob_2 = pred_proba[2] if len(pred_proba) > 2 else 0
                    
                    if prob_1 >= prob_2:
                        pred = "1"
                        confidence = float(prob_1)
                    else:
                        pred = "2"
                        confidence = float(prob_2)
                else:  # lucky7eu
                    # For lucky7eu, predict 0, 1, or 2
                    prob_0 = pred_proba[0] if len(pred_proba) > 0 else 0
                    prob_1 = pred_proba[1] if len(pred_proba) > 1 else 0
                    prob_2 = pred_proba[2] if len(pred_proba) > 2 else 0
                    
                    max_prob = max(prob_0, prob_1, prob_2)
                    if max_prob == prob_0:
                        pred = "0"
                        confidence = float(prob_0)
                    elif max_prob == prob_1:
                        pred = "1"
                        confidence = float(prob_1)
                    else:
                        pred = "2"
                        confidence = float(prob_2)
                
                if confidence < self.min_confidence_threshold:
                    logging.warning(
                        f"Low confidence prediction for {endpoint_type}: {confidence:.2f}. "
                        f"Consider retraining model."
                    )
                
                predictions.append(pred)
                last_sequence = np.append(last_sequence[1:], int(pred))
            
            # Save predictions with confidence scores
            next_mid = str(int(historical_data[0]["mid"]) + 1)
            for i, (pred, confidence) in enumerate(zip(predictions, [float(np.max(self.model.predict_proba([last_sequence])[0])) for _ in range(n_predictions)])):
                pred_mid = str(int(next_mid) + i)
                self.db.save_prediction(pred_mid, pred, current_time, endpoint_type, confidence)
                self.last_predictions[pred_mid] = pred
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error making predictions for {endpoint_type}: {str(e)}")
            return ["0"] * n_predictions

    def generate_predictions(self, endpoint_type: str) -> None:
        """Continuously generate predictions for upcoming rounds."""
        logging.info(f"Starting prediction generation loop for {endpoint_type}")
        while True:
            try:
                # Get the latest data to check for new MID
                results, current_mid, _ = self.fetch_latest_data(endpoint_type)
                if not current_mid:
                    logging.warning(f"No current MID found for {endpoint_type}")
                    time.sleep(1)  # Check every second
                    continue

                # Check if this is a new MID
                last_mid = self.last_mids.get(endpoint_type)
                if last_mid != current_mid:
                    logging.info(f"New MID detected for {endpoint_type}: {current_mid}")
                    self.last_mids[endpoint_type] = current_mid
                    
                    # Generate prediction for new MID
                    self.generate_prediction_for_mid(current_mid, endpoint_type)
                    
                    # Log the prediction details
                    prediction = self.db.get_prediction(current_mid, endpoint_type)
                    if prediction:
                        logging.info(
                            f"Generated and saved prediction for {endpoint_type} - "
                            f"MID: {current_mid}, "
                            f"Value: {prediction['predicted_value']}, "
                            f"Confidence: {prediction['confidence']}"
                        )
                    else:
                        logging.error(f"Failed to save prediction for {endpoint_type} - MID: {current_mid}")
                
                time.sleep(1)  # Check every second
            except Exception as e:
                logging.error(f"Error in prediction generation loop for {endpoint_type}: {str(e)}")
                time.sleep(1)  # Check every second

    def start_prediction_loop(self, endpoint_type: str):
        """Start a prediction generation loop in a separate thread."""
        if endpoint_type in self.prediction_threads and self.prediction_threads[endpoint_type].is_alive():
            logging.info(f"Prediction loop already running for {endpoint_type}")
            return

        def prediction_worker():
            while True:
                try:
                    self.generate_predictions(endpoint_type)
                except Exception as e:
                    logging.error(f"Error in prediction loop for {endpoint_type}: {str(e)}")
                    time.sleep(self.prediction_interval)

        thread = threading.Thread(target=prediction_worker, daemon=True)
        thread.start()
        self.prediction_threads[endpoint_type] = thread
        logging.info(f"Started prediction loop for {endpoint_type}")

    def start_verification_loop(self, endpoint_type: str):
        """Start verification loop in a separate thread."""
        if endpoint_type in self.verification_threads and self.verification_threads[endpoint_type].is_alive():
            logging.info(f"Verification loop already running for {endpoint_type}")
            return

        def verification_worker():
            while True:
                try:
                    self.verify_predictions(endpoint_type)
                except Exception as e:
                    logging.error(f"Error in verification loop for {endpoint_type}: {str(e)}")
                    time.sleep(self.verification_interval)

        # Start prediction loop
        self.start_prediction_loop(endpoint_type)

        # Start verification loop
        thread = threading.Thread(target=verification_worker, daemon=True)
        thread.start()
        self.verification_threads[endpoint_type] = thread
        logging.info(f"Started verification loop for {endpoint_type}")

    def get_current_game_state(self, endpoint_type: str) -> Dict:
        """Get current game state including MID and prediction info."""
        try:
            _, current_mid, _ = self.fetch_latest_data(endpoint_type)
            if not current_mid:
                return {
                    "status": "error",
                    "message": "No active game found",
                    "game_type": endpoint_type
                }

            # Get prediction for current MID
            prediction = self.db.get_prediction(current_mid, endpoint_type)
            prediction_info = None
            if prediction:
                prediction_info = {
                    "predicted_value": prediction["predicted_value"],
                    "confidence": prediction["confidence"],
                    "timestamp": prediction["timestamp"]
                }

            return {
                "status": "success",
                "game_type": endpoint_type,
                "current_mid": current_mid,
                "prediction": prediction_info
            }
        except Exception as e:
            logging.error(f"Error getting game state for {endpoint_type}: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "game_type": endpoint_type
            }
