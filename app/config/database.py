from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

load_dotenv()

class Database:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URL"))
        self.db = self.client[os.getenv("DATABASE_NAME")]
        self.collection = self.db[os.getenv("COLLECTION_NAME")]
        self.prediction_history = self.db[os.getenv("PREDICTION_HISTORY_COLLECTION")]
        self.model_accuracy = self.db[os.getenv("MODEL_ACCURACY_COLLECTION")]
        
        # Create indexes
        self.collection.create_index([("mid", 1), ("endpoint_type", 1)], unique=True)
        self.prediction_history.create_index([("mid", 1), ("endpoint_type", 1)])
        self.prediction_history.create_index("timestamp")
        self.model_accuracy.create_index([("endpoint_type", 1), ("timestamp", -1)])

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

    def save_prediction(self, mid: str, predicted_value: str, timestamp: str, endpoint_type: str) -> Optional[Dict]:
        """Save a new prediction."""
        try:
            return self.prediction_history.insert_one({
                "mid": mid,
                "predicted_value": predicted_value,
                "timestamp": timestamp,
                "endpoint_type": endpoint_type,
                "verified": False,
                "confidence": None  # Will be updated when verified
            })
        except Exception as e:
            logging.error(f"Error saving prediction: {str(e)}")
            return None

    def get_prediction(self, mid: str, endpoint_type: str) -> Optional[Dict]:
        """Get a prediction by MID and endpoint type."""
        return self.prediction_history.find_one({
            "mid": mid,
            "endpoint_type": endpoint_type,
            "verified": False
        })

    def update_prediction_result(self, mid: str, actual_value: str, endpoint_type: str, was_correct: bool) -> bool:
        """Update a prediction with its actual result."""
        try:
            prediction = self.get_prediction(mid, endpoint_type)
            if prediction:
                self.prediction_history.update_one(
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
                return True
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
                        "correct": {"$sum": {"$cond": ["$was_correct", 1, 0]}},
                        "avg_confidence": {"$avg": "$confidence"}
                    }
                }
            ]
            result = list(self.prediction_history.aggregate(pipeline))
            return result[0] if result else {"total": 0, "correct": 0, "avg_confidence": 0}
        except Exception as e:
            logging.error(f"Error getting accuracy metrics: {str(e)}")
            return {"total": 0, "correct": 0, "avg_confidence": 0}

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
                "endpoint_type": endpoint_type
            })
        except Exception as e:
            logging.error(f"Error saving accuracy metrics: {str(e)}")
            return None

    def get_model_performance_history(self, endpoint_type: str, days: int = 7) -> List[Dict]:
        """Get historical model performance data."""
        try:
            return list(self.model_accuracy.find(
                {
                    "endpoint_type": endpoint_type,
                    "timestamp": {
                        "$gte": (datetime.now() - timedelta(days=days)).isoformat()
                    }
                },
                {"_id": 0}
            ).sort("timestamp", 1))
        except Exception as e:
            logging.error(f"Error getting model performance history: {str(e)}")
            return []
