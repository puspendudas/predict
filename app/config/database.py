from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import logging

load_dotenv()

class Database:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URL"))
        self.db = self.client[os.getenv("DATABASE_NAME")]
        self.collection = self.db[os.getenv("COLLECTION_NAME")]
        self.prediction_history = self.db["prediction_history"]
        self.model_accuracy = self.db["model_accuracy"]
        self.prediction_details = self.db["prediction_details"]
        
        # Create indexes
        self._create_indexes()

    def _create_indexes(self):
        """Create necessary indexes for better query performance."""
        try:
            # Results collection
            self.collection.create_index("mid", unique=True)
            self.collection.create_index("timestamp")
            
            # Prediction history
            self.prediction_history.create_index("mid", unique=True)
            self.prediction_history.create_index("timestamp")
            self.prediction_history.create_index("model_version")
            
            # Prediction details
            self.prediction_details.create_index("mid", unique=True)
            self.prediction_details.create_index("timestamp")
            self.prediction_details.create_index("model_version")
            self.prediction_details.create_index("verified")
            
            # Model accuracy
            self.model_accuracy.create_index("timestamp")
            self.model_accuracy.create_index("model_version")
            
        except Exception as e:
            logging.error(f"Error creating indexes: {str(e)}")

    def insert_result(self, data):
        try:
            return self.collection.update_one(
                {"mid": data["mid"]},
                {"$setOnInsert": data},
                upsert=True
            )
        except Exception as e:
            logging.error(f"Error inserting result: {str(e)}")
            return None

    def get_last_n_results(self, n=50):
        return list(self.collection.find({}, {"_id": 0}).sort("mid", -1).limit(n))

    def save_prediction(self, mid: str, predicted_value: str, timestamp: str):
        return self.prediction_history.insert_one({
            "mid": mid,
            "predicted_value": predicted_value,
            "timestamp": timestamp,
            "verified": False
        })

    def save_prediction_details(self, prediction_data: dict) -> bool:
        """Save detailed prediction information."""
        try:
            result = self.prediction_details.update_one(
                {"mid": prediction_data["mid"]},
                {"$set": prediction_data},
                upsert=True
            )
            return result.acknowledged
        except Exception as e:
            logging.error(f"Error saving prediction details: {str(e)}")
            return False

    def get_prediction_details(self, mid: str) -> dict:
        """Retrieve detailed prediction information."""
        try:
            return self.prediction_details.find_one({"mid": mid}, {"_id": 0})
        except Exception as e:
            logging.error(f"Error getting prediction details: {str(e)}")
            return {}

    def update_prediction_result(self, mid: str, actual_result: str) -> bool:
        """Update prediction result and accuracy metrics."""
        try:
            # Get prediction details
            prediction = self.prediction_details.find_one({"mid": mid, "verified": False})
            if not prediction:
                return None
            
            was_correct = prediction["prediction"] == actual_result
            
            # Update prediction details
            self.prediction_details.update_one(
                {"_id": prediction["_id"]},
                {
                    "$set": {
                        "actual_result": actual_result,
                        "verified": True,
                        "was_correct": was_correct,
                        "verification_time": datetime.now().isoformat()
                    }
                }
            )
            
            return was_correct
            
        except Exception as e:
            logging.error(f"Error updating prediction result: {str(e)}")
            return None

    def update_accuracy_metrics(self, metrics_data: dict) -> bool:
        """Update detailed accuracy metrics."""
        try:
            result = self.model_accuracy.insert_one({
                **metrics_data,
                "recorded_at": datetime.now().isoformat()
            })
            return result.acknowledged
        except Exception as e:
            logging.error(f"Error updating accuracy metrics: {str(e)}")
            return False

    def get_accuracy_metrics(self, last_n_days=None, model_version=None):
        """Get detailed accuracy metrics with filtering options."""
        try:
            match_condition = {"verified": True}
            
            if last_n_days is not None:
                match_condition["timestamp"] = {
                    "$gte": (datetime.now() - timedelta(days=last_n_days)).isoformat()
                }
            
            if model_version:
                match_condition["model_version"] = model_version
                
            pipeline = [
                {"$match": match_condition},
                {
                    "$group": {
                        "_id": "$model_version",
                        "total": {"$sum": 1},
                        "correct": {"$sum": {"$cond": ["$was_correct", 1, 0]}},
                        "fallback_total": {"$sum": {"$cond": ["$fallback_used", 1, 0]}},
                        "fallback_correct": {
                            "$sum": {
                                "$cond": [
                                    {"$and": ["$fallback_used", "$was_correct"]},
                                    1,
                                    0
                                ]
                            }
                        },
                        "avg_confidence": {"$avg": "$confidence"}
                    }
                }
            ]
            
            results = list(self.prediction_details.aggregate(pipeline))
            
            if not results:
                return {"total": 0, "correct": 0, "fallback_total": 0, "fallback_correct": 0, "avg_confidence": 0}
            
            return results[0]
            
        except Exception as e:
            logging.error(f"Error getting accuracy metrics: {str(e)}")
            return {"total": 0, "correct": 0, "fallback_total": 0, "fallback_correct": 0, "avg_confidence": 0}

    def save_accuracy_metrics(self, accuracy: float, total_predictions: int):
        try:
            return self.model_accuracy.insert_one({
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "total_predictions": total_predictions
            })
        except Exception as e:
            logging.error(f"Error saving accuracy metrics: {str(e)}")
            return None
