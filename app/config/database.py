from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import redis
import json
import time
from app.config.logging_config import setup_logging

load_dotenv()

# Initialize logging
logger = setup_logging()


class RedisCache:
    """Redis caching layer for predictions and results."""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Connect to Redis with retry logic."""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.client.ping()
                self.connected = True
                logger.info("Successfully connected to Redis")
                return
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.warning("Redis not available, caching disabled")
        self.connected = False
    
    def get(self, key: str) -> Optional[dict]:
        """Get cached value."""
        if not self.connected:
            return None
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: dict, ttl: int = 30) -> bool:
        """Set cached value with TTL in seconds."""
        if not self.connected:
            return False
        try:
            self.client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        if not self.connected:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False


class Database:
    def __init__(self):
        self.cache = RedisCache()
        self._connect_mongodb()
    
    def _connect_mongodb(self):
        """Connect to MongoDB with retry logic for Docker startup."""
        mongodb_url = os.getenv("MONGODB_URL")
        max_retries = 10
        retry_delay = 3
        
        for attempt in range(max_retries):
            try:
                # Connect with settings optimized for internal Docker MongoDB
                self.client = MongoClient(
                    mongodb_url,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=10000,
                    socketTimeoutMS=30000,
                    maxPoolSize=50,
                    minPoolSize=5,
                    retryWrites=True,
                    retryReads=True
                )
                
                # Test the connection
                self.client.server_info()
                
                self.db = self.client[os.getenv("DATABASE_NAME")]
                self.collection = self.db[os.getenv("COLLECTION_NAME")]
                self.prediction_history = self.db[os.getenv("PREDICTION_HISTORY_COLLECTION")]
                self.model_accuracy = self.db[os.getenv("MODEL_ACCURACY_COLLECTION")]
                
                # Create indexes (idempotent)
                self._create_indexes()
                
                logger.info("Successfully connected to MongoDB")
                return
            except Exception as e:
                logger.warning(f"MongoDB connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        raise Exception("Failed to connect to MongoDB after maximum retries")
    
    def _create_indexes(self):
        """Create database indexes for optimal query performance."""
        try:
            # Results collection indexes
            self.collection.create_index(
                [("mid", 1), ("endpoint_type", 1)], 
                unique=True,
                background=True
            )
            self.collection.create_index(
                [("endpoint_type", 1), ("timestamp", -1)],
                background=True
            )
            
            # Prediction history indexes
            self.prediction_history.create_index(
                [("mid", 1), ("endpoint_type", 1)],
                background=True
            )
            self.prediction_history.create_index(
                [("timestamp", -1)],
                background=True
            )
            self.prediction_history.create_index(
                [("endpoint_type", 1), ("verified", 1), ("timestamp", -1)],
                background=True
            )
            
            # Model accuracy indexes
            self.model_accuracy.create_index(
                [("endpoint_type", 1), ("timestamp", -1)],
                background=True
            )
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes (may already exist): {e}")

    def insert_result(self, data: Dict, endpoint_type: str) -> Optional[Dict]:
        """Insert a new result into the database."""
        try:
            data['endpoint_type'] = endpoint_type
            data['timestamp'] = datetime.now().isoformat()
            result = self.collection.update_one(
                {"mid": data["mid"], "endpoint_type": endpoint_type},
                {"$setOnInsert": data},
                upsert=True
            )
            
            # Invalidate cache for this endpoint
            self.cache.delete(f"last_results:{endpoint_type}")
            
            return result
        except Exception as e:
            logger.error(f"Error inserting result: {str(e)}")
            return None

    def get_last_n_results(self, n: int, endpoint_type: str) -> List[Dict]:
        """Get the last n results for a specific endpoint with caching."""
        cache_key = f"last_results:{endpoint_type}:{n}"
        
        # Try cache first for small queries
        if n <= 100:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        results = list(self.collection.find(
            {"endpoint_type": endpoint_type},
            {"_id": 0}
        ).sort("mid", -1).limit(n))
        
        # Cache for 10 seconds
        if n <= 100 and results:
            self.cache.set(cache_key, results, ttl=10)
        
        return results

    def save_prediction(self, mid: str, predicted_value: str, timestamp: str, endpoint_type: str, confidence: float = 0.0) -> Optional[Dict]:
        """Save a new prediction with confidence score."""
        try:
            # Check if prediction already exists
            existing_prediction = self.prediction_history.find_one({
                "mid": mid,
                "endpoint_type": endpoint_type,
                "verified": False
            })
            
            if existing_prediction:
                logger.info(f"Prediction already exists for {endpoint_type} - MID: {mid}")
                return existing_prediction
            
            # Create prediction document with all required fields
            prediction_doc = {
                "mid": mid,
                "predicted_value": predicted_value,
                "timestamp": timestamp,
                "endpoint_type": endpoint_type,
                "verified": False,
                "confidence": confidence,
                "actual_value": None,
                "was_correct": None,
                "verification_timestamp": None
            }
            
            logger.info(f"Saving prediction for {endpoint_type} - MID: {mid}, Value: {predicted_value}, Confidence: {confidence}")
            result = self.prediction_history.insert_one(prediction_doc)
            
            if result.inserted_id:
                # Cache the prediction
                self.cache.set(
                    f"prediction:{endpoint_type}:{mid}",
                    prediction_doc,
                    ttl=300  # 5 minutes
                )
                
                saved_prediction = self.prediction_history.find_one({"_id": result.inserted_id})
                if saved_prediction:
                    logger.info(f"Successfully saved prediction for {endpoint_type} - MID: {mid}")
                    return saved_prediction
                else:
                    logger.error(f"Failed to verify saved prediction for {endpoint_type} - MID: {mid}")
                    return None
            else:
                logger.error(f"Failed to save prediction for {endpoint_type} - MID: {mid}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return None

    def get_prediction(self, mid: str, endpoint_type: str) -> Optional[Dict]:
        """Get a prediction by MID and endpoint type with caching."""
        cache_key = f"prediction:{endpoint_type}:{mid}"
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # First try to get unverified prediction
            prediction = self.prediction_history.find_one({
                "mid": mid,
                "endpoint_type": endpoint_type,
                "verified": False
            })
            
            if not prediction:
                # If no unverified prediction, try to get the most recent verified prediction
                prediction = self.prediction_history.find_one({
                    "mid": mid,
                    "endpoint_type": endpoint_type,
                    "verified": True
                }, sort=[("timestamp", -1)])
            
            if prediction:
                # Convert ObjectId for caching
                prediction_copy = {k: v for k, v in prediction.items() if k != '_id'}
                self.cache.set(cache_key, prediction_copy, ttl=60)
                logger.info(f"Found prediction for {endpoint_type} - MID: {mid}")
            else:
                logger.info(f"No prediction found for {endpoint_type} - MID: {mid}")
            return prediction
        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
            return None

    def update_prediction_result(self, mid: str, actual_value: str, endpoint_type: str, was_correct: bool) -> bool:
        """Update prediction with actual result and mark as verified."""
        try:
            # Find the prediction
            prediction = self.prediction_history.find_one({
                "mid": mid,
                "endpoint_type": endpoint_type,
                "verified": False
            })
            
            if prediction:
                # Log the prediction we found
                logger.info(
                    f"Found prediction to update for {endpoint_type} - "
                    f"MID: {mid}, "
                    f"Current Value: {prediction.get('predicted_value')}, "
                    f"Actual Value: {actual_value}, "
                    f"Was Correct: {was_correct}"
                )
                
                # Update the prediction document with all verification fields
                current_time = datetime.now().isoformat()
                update_fields = {
                    "actual_value": actual_value,
                    "verified": True,
                    "was_correct": was_correct,
                    "verification_timestamp": current_time
                }
                
                # Log the update fields for debugging
                logger.info(f"Updating prediction fields for {endpoint_type} - MID: {mid}: {update_fields}")
                
                # Update the prediction document
                update_result = self.prediction_history.update_one(
                    {"_id": prediction["_id"]},
                    {"$set": update_fields}
                )
                
                if update_result.modified_count == 0:
                    logger.error(
                        f"Failed to update prediction in database - MID: {mid}, "
                        f"Endpoint: {endpoint_type}"
                    )
                    return False
                
                # Invalidate cache
                self.cache.delete(f"prediction:{endpoint_type}:{mid}")
                self.cache.delete(f"accuracy:{endpoint_type}")
                
                # Verify the update
                updated_prediction = self.prediction_history.find_one({"_id": prediction["_id"]})
                if updated_prediction and updated_prediction.get("verified"):
                    logger.info(
                        f"Successfully updated prediction in database - MID: {mid}, "
                        f"Endpoint: {endpoint_type}, "
                        f"Actual Value: {actual_value}, "
                        f"Was Correct: {was_correct}, "
                        f"Verification Time: {current_time}"
                    )
                else:
                    logger.error(
                        f"Prediction update verification failed - MID: {mid}, "
                        f"Endpoint: {endpoint_type}"
                    )
                    return False
                
                # Update accuracy metrics immediately
                metrics = self.get_accuracy_metrics(endpoint_type, last_n_days=1)
                if metrics["total"] > 0:
                    accuracy = metrics["correct"] / metrics["total"]
                    self.save_accuracy_metrics(
                        accuracy=accuracy,
                        total_predictions=metrics["total"],
                        endpoint_type=endpoint_type
                    )
                    logger.info(f"Updated accuracy metrics for {endpoint_type}: {accuracy:.2f}")
                
                return True
                
            logger.warning(
                f"No unverified prediction found to update - MID: {mid}, "
                f"Endpoint: {endpoint_type}"
            )
            return False
        except Exception as e:
            logger.error(f"Error updating prediction result: {str(e)}")
            return False

    def get_accuracy_metrics(self, endpoint_type: str, last_n_days: Optional[int] = 7) -> Dict:
        """Get accuracy metrics for a specific endpoint with caching."""
        cache_key = f"accuracy:{endpoint_type}:{last_n_days}"
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
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
                logger.info(
                    f"Accuracy metrics for {endpoint_type}: "
                    f"Total: {metrics['total']}, "
                    f"Correct: {metrics['correct']}, "
                    f"Accuracy: {metrics['accuracy']:.2f}"
                )
                
                # Cache for 30 seconds
                self.cache.set(cache_key, metrics, ttl=30)
                return metrics
                
            logger.warning(f"No accuracy metrics found for {endpoint_type}")
            default_metrics = {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0, "avg_confidence": 0}
            return default_metrics
        except Exception as e:
            logger.error(f"Error getting accuracy metrics: {str(e)}")
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
            logger.error(f"Error getting consecutive incorrect predictions: {str(e)}")
            return 0

    def save_accuracy_metrics(self, accuracy: float, total_predictions: int, endpoint_type: str) -> Optional[Dict]:
        """Save accuracy metrics for a specific endpoint."""
        try:
            result = self.model_accuracy.insert_one({
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "total_predictions": total_predictions,
                "endpoint_type": endpoint_type,
                "model_version": "2.0",  # Updated model version
                "training_samples": total_predictions,
                "last_updated": datetime.now().isoformat()
            })
            
            # Invalidate accuracy cache
            self.cache.delete(f"accuracy:{endpoint_type}")
            
            return result
        except Exception as e:
            logger.error(f"Error saving accuracy metrics: {str(e)}")
            return None

    def get_model_performance_history(self, endpoint_type: str, limit: int = 100) -> List[Dict]:
        """Get historical performance data for the model."""
        try:
            return list(self.model_accuracy.find(
                {"endpoint_type": endpoint_type},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit))
        except Exception as e:
            logger.error(f"Error getting model performance history: {str(e)}")
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
            logger.error(f"Error getting recent accuracy trend: {str(e)}")
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
            logger.error(f"Error getting prediction history: {str(e)}")
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
            logger.error(f"Error getting results by date range: {str(e)}")
            return []

    def get_latest_result(self, endpoint_type: str) -> Optional[Dict]:
        """Get the most recent result for a specific endpoint type."""
        try:
            result = self.collection.find_one(
                {"endpoint_type": endpoint_type},
                sort=[("timestamp", -1)]
            )
            if result:
                logger.info(f"Found latest result for {endpoint_type} - MID: {result.get('mid')}")
            else:
                logger.info(f"No results found for {endpoint_type}")
            return result
        except Exception as e:
            logger.error(f"Error getting latest result: {str(e)}")
            return None

    def get_predictions_by_mids(self, mids: List[str], endpoint_type: str) -> List[Dict]:
        """Get predictions for specific MIDs and endpoint type."""
        try:
            predictions = list(self.prediction_history.find({
                "mid": {"$in": mids},
                "endpoint_type": endpoint_type,
                "verified": False
            }))
            if predictions:
                logger.info(f"Found {len(predictions)} predictions for {endpoint_type} MIDs: {mids}")
            else:
                logger.info(f"No predictions found for {endpoint_type} MIDs: {mids}")
            return predictions
        except Exception as e:
            logger.error(f"Error getting predictions by MIDs: {str(e)}")
            return []
