import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import logging
import os
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def optimize_rf_params(X=None, y=None):
    """Get optimized RandomForest parameters."""
    if X is not None and y is not None:
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 15, 20],
            'min_samples_split': [4, 5, 6],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        return grid_search.best_params_
    
    return {
        'n_estimators': 400,
        'max_depth': 15,
        'min_samples_split': 5,
        'class_weight': 'balanced',
        'random_state': 42
    }

def initialize_models(X=None, y=None):
    """Initialize enhanced ensemble model with multiple algorithms."""
    # Get optimized parameters
    rf_params = optimize_rf_params(X, y)
    
    # Base models
    rf_model = RandomForestClassifier(**rf_params)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=12,
        subsample=0.8,
        random_state=42
    )
    
    et_model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        class_weight='balanced',
        random_state=42
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    ada_model = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('et', et_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('ada', ada_model)
        ],
        voting='soft',
        weights=[2, 1, 1, 2, 2, 1]  # Give more weight to RF, XGBoost, and LightGBM
    )
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    
    return ensemble, scaler

def validate_model(ensemble, scaler, X, y):
    """Validate model performance using cross-validation."""
    try:
        # Scale the features
        X_scaled = scaler.fit_transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring='accuracy')
        
        # Get detailed metrics using a single split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
        
        metrics = {
            'cv_scores_mean': float(cv_scores.mean()),
            'cv_scores_std': float(cv_scores.std()),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
        return metrics, True
        
    except Exception as e:
        logging.error(f"Error in model validation: {str(e)}")
        return None, False

def save_models(ensemble, scaler, metrics=None, model_dir="./ml_model"):
    """Save models and metadata to disk."""
    try:
        # Create directory if it doesn't exist
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        model_path = model_dir / "prediction_model.pkl"
        scaler_path = model_dir / "scaler.pkl"
        metadata_path = model_dir / "model_metadata.json"
        
        joblib.dump(ensemble, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'ensemble_estimators': [est[0] for est in ensemble.estimators],
            'ensemble_weights': ensemble.weights,
            'metrics': metrics if metrics else {}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logging.info(f"Models and metadata saved successfully to {model_dir}")
        logging.info(f"Model path: {model_path}")
        logging.info(f"Scaler path: {scaler_path}")
        logging.info(f"Metadata path: {metadata_path}")
        
        if metrics:
            logging.info("Model Performance Metrics:")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")
        
    except Exception as e:
        logging.error(f"Error saving models: {str(e)}")
        raise

def generate_sample_data():
    """Generate sample data for initial model training."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sequence-like features
    X = np.random.rand(n_samples, 30)  # 30 features
    # Generate balanced classes (0, 1, 2)
    y = np.random.randint(0, 3, n_samples)
    
    return X, y

def main():
    try:
        logging.info("Generating sample data for initial training...")
        X, y = generate_sample_data()
        
        logging.info("Initializing models with optimization...")
        ensemble, scaler = initialize_models(X, y)
        
        logging.info("Validating models...")
        metrics, is_valid = validate_model(ensemble, scaler, X, y)
        
        if is_valid:
            logging.info("Saving models and metadata...")
            save_models(ensemble, scaler, metrics)
            logging.info("Model generation completed successfully!")
        else:
            logging.error("Model validation failed. Models not saved.")
            
    except Exception as e:
        logging.error(f"Error generating models: {str(e)}")
        raise

if __name__ == "__main__":
    main() 