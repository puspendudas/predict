import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import json
import os
from .config.database import Database
from .advanced_predictor import AdvancedEnsemblePredictor
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import time
import urllib3
import certifi
import asyncio
import threading
import httpx
from app.config.logging_config import setup_logging

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize logging
logger = setup_logging()


class PredictionService:
    def __init__(self):
        self.db = Database()
        self.last_predictions = {}
        self.last_mids = {}  # Store last seen MID for each game type
        
        # Configuration from environment
        self.sequence_length = int(os.getenv("SEQUENCE_LENGTH", "20"))
        self.max_samples_for_training = int(os.getenv("MAX_TRAINING_SAMPLES", "2000"))
        self.min_samples_for_training = int(os.getenv("MIN_TRAINING_SAMPLES", "50"))
        self.min_confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
        self.accuracy_threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.6"))
        
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
        
        self.verification_interval = 5  # seconds
        self.prediction_interval = 30  # seconds
        self.prediction_window = 2
        self.verification_threads = {}
        self.prediction_threads = {}
        
        # Initialize advanced ensemble predictors for each game type
        self.ensemble_predictors = {
            'teen20': AdvancedEnsemblePredictor('teen20'),
            'lucky7eu': AdvancedEnsemblePredictor('lucky7eu'),
            'dt20': AdvancedEnsemblePredictor('dt20')
        }
        
        # HTTP client for faster requests
        self.http_client = None
        self._init_http_client()
    
    def _init_http_client(self):
        """Initialize async HTTP client with connection pooling."""
        try:
            # Use a shared session for connection pooling
            self.session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=3
            )
            self.session.mount('https://', adapter)
            self.session.mount('http://', adapter)
            logger.info("HTTP client initialized with connection pooling")
        except Exception as e:
            logger.error(f"Error initializing HTTP client: {e}")
            self.session = requests.Session()

    def fetch_latest_data(self, endpoint_type='teen20') -> Tuple[List[Dict], Optional[str], Optional[Dict]]:
        """Fetch latest data and return results, current MID, and game info."""
        try:
            # Get current MID and game info from odds API
            odds_url = self.endpoints.get(endpoint_type)
            if not odds_url:
                raise ValueError(f"Invalid endpoint type: {endpoint_type}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json',
                'Origin': 'https://terminal.hpterminal.com',
                'Referer': 'https://terminal.hpterminal.com/',
            }
            
            try:
                response = self.session.get(odds_url, headers=headers, verify=certifi.where(), timeout=5)
                response.raise_for_status()
                odds_data = response.json()
            except requests.exceptions.SSLError:
                response = self.session.get(odds_url, headers=headers, verify=False, timeout=5)
                response.raise_for_status()
                odds_data = response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching odds data for {endpoint_type}: {str(e)}")
                return [], None, None
            
            # Log the full response for debugging
            logging.debug(f"Odds API response for {endpoint_type}: {json.dumps(odds_data, indent=2)}")
            
            # Get current MID and game info from t1
            current_mid = None
            game_info = None
            
            # Try different possible paths to get t1 data
            t1_data = None
            if "data" in odds_data:
                if isinstance(odds_data["data"], dict):
                    if "data" in odds_data["data"]:
                        if isinstance(odds_data["data"]["data"], dict):
                            if "data" in odds_data["data"]["data"]:
                                if isinstance(odds_data["data"]["data"]["data"], dict):
                                    t1_data = odds_data["data"]["data"]["data"].get("t1", [])
                                else:
                                    t1_data = odds_data["data"]["data"].get("t1", [])
                            else:
                                t1_data = odds_data["data"]["data"].get("t1", [])
                        else:
                            t1_data = odds_data["data"].get("t1", [])
                    else:
                        t1_data = odds_data["data"].get("t1", [])
                else:
                    t1_data = odds_data.get("t1", [])
            
            if t1_data and len(t1_data) > 0:
                t1_data = t1_data[0]
                current_mid = t1_data.get("mid")
                if current_mid:
                    logging.debug(f"Found current MID for {endpoint_type}: {current_mid}")
                else:
                    logging.warning(f"No MID found in t1 data for {endpoint_type}")
                
                game_info = {
                    "mid": current_mid,
                    "autotime": t1_data.get("autotime"),
                    "gtype": t1_data.get("gtype"),
                    "max": t1_data.get("max"),
                    "min": t1_data.get("min")
                }
            else:
                logging.warning(f"No t1 data found in response for {endpoint_type}")
            
            # Get results from results API
            results_url = self.result_endpoints.get(endpoint_type)
            if not results_url:
                raise ValueError(f"Invalid results endpoint type: {endpoint_type}")
            
            try:
                response = self.session.get(results_url, headers=headers, verify=certifi.where(), timeout=5)
                response.raise_for_status()
                results_data = response.json()
            except requests.exceptions.SSLError:
                response = self.session.get(results_url, headers=headers, verify=False, timeout=5)
                response.raise_for_status()
                results_data = response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching results data for {endpoint_type}: {str(e)}")
                return [], current_mid, game_info
            
            # Get results
            results = []
            if "data" in results_data:
                if isinstance(results_data["data"], dict):
                    if "data" in results_data["data"]:
                        if isinstance(results_data["data"]["data"], dict):
                            if "data" in results_data["data"]["data"]:
                                if isinstance(results_data["data"]["data"]["data"], dict):
                                    results = results_data["data"]["data"]["data"].get("result", [])
                                else:
                                    results = results_data["data"]["data"].get("result", [])
                            else:
                                results = results_data["data"]["data"].get("result", [])
                        else:
                            results = results_data["data"].get("result", [])
                    else:
                        results = results_data["data"].get("result", [])
                else:
                    results = results_data.get("result", [])
            
            if results:
                logging.debug(f"Got {len(results)} results for {endpoint_type}")
            
            return results, current_mid, game_info
        except Exception as e:
            logging.error(f"Error in fetch_latest_data for {endpoint_type}: {str(e)}")
            return [], None, None

    def verify_predictions(self, endpoint_type: str) -> None:
        """Verify predictions against actual results every second."""
        logging.info(f"Starting verification loop for {endpoint_type}")
        while True:
            try:
                # Get latest results from casino-last-10-results API
                results, current_mid, _ = self.fetch_latest_data(endpoint_type)
                if not results:
                    logging.warning(f"No results found for {endpoint_type}")
                    time.sleep(1)
                    continue

                # Process each result
                for result in results:
                    result_mid = result["mid"]
                    actual_value = result["result"]
                    
                    # Find unverified prediction for this MID
                    prediction = self.db.prediction_history.find_one({
                        "mid": result_mid,
                        "endpoint_type": endpoint_type,
                        "verified": False
                    })
                    
                    if prediction:
                        predicted_value = prediction["predicted_value"]
                        was_correct = actual_value == predicted_value
                        
                        logging.info(
                            f"Verifying {endpoint_type} - MID: {result_mid}, "
                            f"Predicted: {predicted_value}, Actual: {actual_value}, "
                            f"Correct: {was_correct}"
                        )
                        
                        # Update prediction with actual result
                        update_success = self.db.update_prediction_result(
                            result_mid,
                            actual_value,
                            endpoint_type,
                            was_correct
                        )
                        
                        if update_success:
                            # Save the actual result
                            self.db.insert_result(result, endpoint_type)
                            
                            # Check if model needs retraining
                            self.check_and_update_model(endpoint_type)
                
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in verification loop for {endpoint_type}: {str(e)}")
                time.sleep(1)

    def generate_prediction_for_mid(self, mid: str, endpoint_type: str) -> None:
        """Generate prediction for a specific MID using ensemble predictor."""
        try:
            # Get historical data for prediction
            historical_data = self.db.get_last_n_results(self.max_samples_for_training, endpoint_type)
            if len(historical_data) < self.min_samples_for_training:
                logging.warning(f"Insufficient historical data for {endpoint_type}: {len(historical_data)} samples")
                return
            
            predictor = self.ensemble_predictors[endpoint_type]
            
            # Ensure model is fitted
            if not predictor.fitted:
                logging.info(f"Training ensemble model for {endpoint_type} with {len(historical_data)} samples")
                accuracy = predictor.fit(historical_data)
                logging.info(f"Ensemble model trained for {endpoint_type} with accuracy: {accuracy:.4f}")
            
            # Prepare the latest sequence for prediction
            results = [int(d["result"]) for d in historical_data[:self.sequence_length]]
            last_sequence = np.array(results[::-1])  # Reverse to get chronological order
            
            # Generate prediction using game-specific strategy
            pred, confidence = predictor.predict_with_strategy(last_sequence, endpoint_type)
            
            logging.info(f"Prediction for {endpoint_type}: {pred} (confidence: {confidence:.4f})")
            
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
                    f"MID: {mid}, Value: {pred}, Confidence: {confidence:.4f}"
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
        """Update the ensemble model with recent data."""
        try:
            # Get historical data for training
            historical_data = self.db.get_last_n_results(
                self.max_samples_for_training,
                endpoint_type
            )
            
            if len(historical_data) < self.min_samples_for_training:
                logging.warning(f"Insufficient data for model update: {len(historical_data)} samples")
                return
            
            predictor = self.ensemble_predictors[endpoint_type]
            
            # Retrain the ensemble
            accuracy = predictor.fit(historical_data)
            
            logging.info(
                f"Ensemble model updated for {endpoint_type} with {len(historical_data)} samples. "
                f"New accuracy: {accuracy:.4f}"
            )
            
            # Save the accuracy metrics
            self.db.save_accuracy_metrics(
                accuracy=accuracy,
                total_predictions=len(historical_data),
                endpoint_type=endpoint_type
            )
            
        except Exception as e:
            logging.error(f"Error updating model for {endpoint_type}: {str(e)}")

    def predict_next_rounds(self, endpoint_type='teen20', n_predictions=2) -> List[str]:
        """Generate predictions for the next rounds."""
        try:
            historical_data = self.db.get_last_n_results(self.max_samples_for_training, endpoint_type)
            if len(historical_data) < self.min_samples_for_training:
                logging.warning(f"Insufficient historical data for {endpoint_type}: {len(historical_data)} samples")
                return ["0"] * n_predictions
            
            predictor = self.ensemble_predictors[endpoint_type]
            
            # Ensure model is fitted
            if not predictor.fitted:
                logging.info(f"Training ensemble model for {endpoint_type} with {len(historical_data)} samples")
                predictor.fit(historical_data)
            
            current_time = datetime.now().isoformat()
            results = [int(d["result"]) for d in historical_data[:self.sequence_length]]
            last_sequence = np.array(results[::-1])
            
            predictions = []
            for _ in range(n_predictions):
                pred, confidence = predictor.predict_with_strategy(last_sequence, endpoint_type)
                predictions.append(pred)
                # Update sequence for next prediction
                last_sequence = np.append(last_sequence[1:], int(pred))
            
            # Save predictions with confidence scores
            next_mid = str(int(historical_data[0]["mid"]) + 1)
            for i, pred in enumerate(predictions):
                pred_mid = str(int(next_mid) + i)
                self.db.save_prediction(pred_mid, pred, current_time, endpoint_type, 0.5)
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
                    time.sleep(1)
                    continue

                # Check if this is a new MID
                last_mid = self.last_mids.get(endpoint_type)
                if last_mid != current_mid:
                    logging.info(f"New MID detected for {endpoint_type}: {current_mid}")
                    self.last_mids[endpoint_type] = current_mid
                    
                    # Generate prediction for new MID
                    self.generate_prediction_for_mid(current_mid, endpoint_type)
                
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in prediction generation loop for {endpoint_type}: {str(e)}")
                time.sleep(1)

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
