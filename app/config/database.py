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
        self.prediction_history = self.db[os.getenv("PREDICTION_HISTORY_COLLECTION")]
        self.model_accuracy = self.db[os.getenv("MODEL_ACCURACY_COLLECTION")]
        
        # Create indexes
        self.collection.create_index("mid", unique=True)
        self.prediction_history.create_index("mid")
        self.prediction_history.create_index("timestamp")

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

    def update_prediction_result(self, mid: str, actual_value: str):
        prediction = self.prediction_history.find_one({"mid": mid, "verified": False})
        if prediction:
            was_correct = prediction["predicted_value"] == actual_value
            self.prediction_history.update_one(
                {"_id": prediction["_id"]},
                {
                    "$set": {
                        "actual_value": actual_value,
                        "verified": True,
                        "was_correct": was_correct
                    }
                }
            )
            return was_correct
        return None

    def get_accuracy_metrics(self, last_n_days=7):
        try:
            match_condition = {
                "verified": True
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
                        "correct": {"$sum": {"$cond": ["$was_correct", 1, 0]}}
                    }
                }
            ]
            result = list(self.prediction_history.aggregate(pipeline))
            return result[0] if result else {"total": 0, "correct": 0}
        except Exception as e:
            logging.error(f"Error getting accuracy metrics: {str(e)}")
            return {"total": 0, "correct": 0}

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
