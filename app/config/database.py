from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import certifi
import ssl

load_dotenv()

class Database:
    def __init__(self):
        try:
            # Connect with simplified SSL settings
            self.client = MongoClient(
                os.getenv("MONGODB_URL"),
                tlsCAFile=certifi.where(),
                tlsAllowInvalidCertificates=True  # Temporarily allow invalid certificates for testing
            )
            
            # Test the connection
            self.client.server_info()
            
            self.db = self.client[os.getenv("DATABASE_NAME")]
            self.collection = self.db[os.getenv("COLLECTION_NAME")]
            self.prediction_history = self.db[os.getenv("PREDICTION_HISTORY_COLLECTION")]
            self.model_accuracy = self.db[os.getenv("MODEL_ACCURACY_COLLECTION")]
            
            # Create indexes
            self.collection.create_index([("mid", 1), ("endpoint_type", 1)], unique=True)
            self.prediction_history.create_index([("mid", 1), ("endpoint_type", 1)])
            self.prediction_history.create_index("timestamp")
            self.model_accuracy.create_index([("endpoint_type", 1), ("timestamp", -1)])
            
            logging.info("Successfully connected to MongoDB")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def insert_result(self, data: Dict, endpoint_type: str) -> Optional[Dict]:
        """Insert a new result into the database."""
        try:
            data['endpoint_type'] = endpoint_type
            data['timestamp'] = datetime.now().isoformat()
            return self.collection.update_one(
                {"mid": data["mid"], "endpoint_type": endpoint_type},
                {"$setOnInsert": data},
                upsert=True
            )
        except Exception as e:
            logging.error(f"Error inserting result: {str(e)}")
            return None

    def get_last_n_results(self, n: int, endpoint_type: str) -> List[Dict]:
        """Get the last n results for a specific endpoint."""
        return list(self.collection.find(
            {"endpoint_type": endpoint_type},
            {"_id": 0}
        ).sort("mid", -1).limit(n))

    def save_prediction(self, mid: str, predicted_value: str, timestamp: str, endpoint_type: str, confidence: float = 0.0) -> Optional[Dict]:
        """Save a new prediction with confidence score."""
        try:
            logging.info(f"Saving prediction for {endpoint_type} - MID: {mid}, Value: {predicted_value}, Confidence: {confidence}")
            result = self.prediction_history.insert_one({
                "mid": mid,
                "predicted_value": predicted_value,
                "timestamp": timestamp,
                "endpoint_type": endpoint_type,
                "verified": False,
                "confidence": confidence,
                "verification_timestamp": None,
                "was_correct": None
            })
            if result.inserted_id:
                logging.info(f"Successfully saved prediction for {endpoint_type} - MID: {mid}")
                return result
            else:
                logging.error(f"Failed to save prediction for {endpoint_type} - MID: {mid}")
                return None
        except Exception as e:
            logging.error(f"Error saving prediction: {str(e)}")
            return None

    def get_prediction(self, mid: str, endpoint_type: str) -> Optional[Dict]:
        """Get a prediction by MID and endpoint type."""
        try:
            prediction = self.prediction_history.find_one({
                "mid": mid,
                "endpoint_type": endpoint_type,
                "verified": False
            })
            if prediction:
                logging.info(f"Found unverified prediction for {endpoint_type} - MID: {mid}")
            else:
                logging.info(f"No unverified prediction found for {endpoint_type} - MID: {mid}")
            return prediction
        except Exception as e:
            logging.error(f"Error getting prediction: {str(e)}")
            return None

    def update_prediction_result(self, mid: str, actual_value: str, endpoint_type: str, was_correct: bool) -> bool:
        """Update a prediction with its actual result."""
        try:
            prediction = self.get_prediction(mid, endpoint_type)
            if prediction:
                logging.info(
                    f"Updating prediction in database - MID: {mid}, "
                    f"Endpoint: {endpoint_type}, "
                    f"Was Correct: {was_correct}"
                )
                
                update_result = self.prediction_history.update_one(
                    {"_id": prediction["_id"]},
                    {
                        "$set": {
                            "actual_value": actual_value,
                            "verified": True,
                            "was_correct": was_correct,
                            "verification_timestamp": datetime.now().isoformat()
                        }
                    }
                )
                
                if update_result.modified_count == 0:
                    logging.error(
                        f"Failed to update prediction in database - MID: {mid}, "
                        f"Endpoint: {endpoint_type}"
                    )
                    return False
                
                logging.info(
                    f"Successfully updated prediction in database - MID: {mid}, "
                    f"Endpoint: {endpoint_type}, "
                    f"Modified count: {update_result.modified_count}"
                )
                
                # Update accuracy metrics immediately
                metrics = self.get_accuracy_metrics(endpoint_type, last_n_days=1)
                if metrics["total"] > 0:
                    accuracy = metrics["correct"] / metrics["total"]
                    self.save_accuracy_metrics(
                        accuracy=accuracy,
                        total_predictions=metrics["total"],
                        endpoint_type=endpoint_type
                    )
                    logging.info(f"Updated accuracy metrics for {endpoint_type}: {accuracy:.2f}")
                
                return True
                
            logging.warning(
                f"No unverified prediction found to update - MID: {mid}, "
                f"Endpoint: {endpoint_type}"
            )
            return False
        except Exception as e:
            logging.error(f"Error updating prediction result: {str(e)}")
            return False

    def get_accuracy_metrics(self, endpoint_type: str, last_n_days: Optional[int] = 7) -> Dict:
        """Get accuracy metrics for a specific endpoint."""
        try:
            match_condition = {
                "verified": True,
                "endpoint_type": endpoint_type
            }
            
            if last_n_days is not None:
                match_condition["timestamp"] = {
                    "$gte": (datetime.now() - timedelta(days=last_n_days)).isoformat()
                }
                
            pipeline = [
                {"$match": match_condition},
                {
                    "$group": {
                        "_id": None,
                        "total": {"$sum": 1},
                        "correct": {"$sum": {"$cond": [{"$eq": ["$was_correct", True]}, 1, 0]}},
                        "incorrect": {"$sum": {"$cond": [{"$eq": ["$was_correct", False]}, 1, 0]}},
                        "avg_confidence": {"$avg": "$confidence"}
                    }
                }
            ]
            result = list(self.prediction_history.aggregate(pipeline))
            if result:
                metrics = result[0]
                metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
                return metrics
            return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0, "avg_confidence": 0}
        except Exception as e:
            logging.error(f"Error getting accuracy metrics: {str(e)}")
            return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0, "avg_confidence": 0}

    def get_consecutive_incorrect_predictions(self, endpoint_type: str) -> int:
        """Get the number of consecutive incorrect predictions."""
        try:
            pipeline = [
                {
                    "$match": {
                        "endpoint_type": endpoint_type,
                        "verified": True
                    }
                },
                {"$sort": {"timestamp": -1}},
                {
                    "$group": {
                        "_id": None,
                        "consecutive_incorrect": {
                            "$sum": {
                                "$cond": [
                                    {"$eq": ["$was_correct", False]},
                                    1,
                                    {"$cond": [{"$eq": ["$was_correct", True]}, -999999, 0]}
                                ]
                            }
                        }
                    }
                }
            ]
            result = list(self.prediction_history.aggregate(pipeline))
            return max(0, result[0]["consecutive_incorrect"]) if result else 0
        except Exception as e:
            logging.error(f"Error getting consecutive incorrect predictions: {str(e)}")
            return 0

    def save_accuracy_metrics(self, accuracy: float, total_predictions: int, endpoint_type: str) -> Optional[Dict]:
        """Save accuracy metrics for a specific endpoint."""
        try:
            return self.model_accuracy.insert_one({
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "total_predictions": total_predictions,
                "endpoint_type": endpoint_type,
                "model_version": "1.0",  # Track model versions
                "training_samples": total_predictions,
                "last_update": datetime.now().isoformat()
            })
        except Exception as e:
            logging.error(f"Error saving accuracy metrics: {str(e)}")
            return None

    def get_model_performance_history(self, endpoint_type: str, limit: int = 100) -> List[Dict]:
        """Get historical performance data for the model."""
        try:
            return list(self.model_accuracy.find(
                {"endpoint_type": endpoint_type},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit))
        except Exception as e:
            logging.error(f"Error getting model performance history: {str(e)}")
            return []

    def get_recent_accuracy_trend(self, endpoint_type: str, days: int = 7) -> Dict:
        """Get recent accuracy trend for model improvement decisions."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            pipeline = [
                {
                    "$match": {
                        "endpoint_type": endpoint_type,
                        "timestamp": {"$gte": cutoff_date.isoformat()}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "avg_accuracy": {"$avg": "$accuracy"},
                        "min_accuracy": {"$min": "$accuracy"},
                        "max_accuracy": {"$max": "$accuracy"},
                        "total_predictions": {"$sum": "$total_predictions"},
                        "samples": {"$sum": 1}
                    }
                }
            ]
            result = list(self.model_accuracy.aggregate(pipeline))
            return result[0] if result else {
                "avg_accuracy": 0,
                "min_accuracy": 0,
                "max_accuracy": 0,
                "total_predictions": 0,
                "samples": 0
            }
        except Exception as e:
            logging.error(f"Error getting recent accuracy trend: {str(e)}")
            return {
                "avg_accuracy": 0,
                "min_accuracy": 0,
                "max_accuracy": 0,
                "total_predictions": 0,
                "samples": 0
            }

    def get_prediction_history(self, endpoint_type: str, limit: int = 100) -> List[Dict]:
        """Get detailed prediction history for analysis."""
        try:
            return list(self.prediction_history.find(
                {
                    "endpoint_type": endpoint_type,
                    "verified": True
                },
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit))
        except Exception as e:
            logging.error(f"Error getting prediction history: {str(e)}")
            return []

    def get_results_by_date_range(self, endpoint_type: str, start_date: str, end_date: str) -> List[Dict]:
        """Get results for a specific endpoint within a date range."""
        try:
            return list(self.collection.find(
                {
                    "endpoint_type": endpoint_type,
                    "timestamp": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                },
                {"_id": 0}
            ).sort("timestamp", 1))
        except Exception as e:
            logging.error(f"Error getting results by date range: {str(e)}")
            return []
