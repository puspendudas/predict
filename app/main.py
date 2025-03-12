from fastapi import FastAPI
from .predict import PredictionService
from .models import PredictionResponse, ModelAccuracy
import asyncio
import logging
from datetime import datetime

app = FastAPI()
prediction_service = PredictionService()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_data_cron():
    while True:
        try:
            results = prediction_service.fetch_latest_data()
            if results:
                logger.info(f"Data fetch cron: Got {len(results)} results at {datetime.now().isoformat()}")
        except Exception as e:
            logger.error(f"Data fetch cron error: {str(e)}")
        await asyncio.sleep(25)  # Run every 25 seconds

async def accuracy_check_cron():
    while True:
        try:
            metrics = prediction_service.check_and_update_accuracy()
            
            # Save accuracy metrics to database
            prediction_service.db.save_accuracy_metrics(
                accuracy=metrics['accuracy'],
                total_predictions=metrics['total_predictions']
            )
            
            logger.info(
                f"Accuracy check cron: Current accuracy {metrics['accuracy']:.2f} " 
                f"(Total: {metrics['total_predictions']}, Correct: {metrics['correct_predictions']}) "
                f"at {datetime.now().isoformat()}"
            )
        except Exception as e:
            logger.error(f"Accuracy check cron error: {str(e)}")
        await asyncio.sleep(300)  # Run every 5 minutes

@app.on_event("startup")
async def startup_event():
    # Start both cron tasks
    asyncio.create_task(fetch_data_cron())
    asyncio.create_task(accuracy_check_cron())

@app.get("/predict", response_model=PredictionResponse)
async def get_prediction():
    current_results = prediction_service.fetch_latest_data()
    predictions = prediction_service.predict_next_rounds()
    return PredictionResponse(
        current_results=current_results,
        predictions=predictions
    )

@app.get("/accuracy", response_model=ModelAccuracy)
async def get_model_accuracy():
    metrics = prediction_service.db.get_accuracy_metrics()
    return ModelAccuracy(
        timestamp=datetime.now().isoformat(),
        accuracy=metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0,
        total_predictions=metrics["total"]
    )
