// MongoDB initialization script
// Creates the prediction database and necessary collections with indexes

db = db.getSiblingDB('prediction_db');

// Create collections
db.createCollection('results');
db.createCollection('prediction_history');
db.createCollection('model_accuracy');

// Create indexes for results collection
db.results.createIndex({ "mid": 1, "endpoint_type": 1 }, { unique: true });
db.results.createIndex({ "endpoint_type": 1, "timestamp": -1 });

// Create indexes for prediction_history collection
db.prediction_history.createIndex({ "mid": 1, "endpoint_type": 1 });
db.prediction_history.createIndex({ "timestamp": -1 });
db.prediction_history.createIndex({ "endpoint_type": 1, "verified": 1 });
db.prediction_history.createIndex({ "endpoint_type": 1, "verified": 1, "timestamp": -1 });

// Create indexes for model_accuracy collection
db.model_accuracy.createIndex({ "endpoint_type": 1, "timestamp": -1 });

print('MongoDB initialized successfully with collections and indexes');
