from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.predict import PredictionService
from app.models import PredictionResponse, ModelAccuracy
import asyncio
import logging
from datetime import datetime
from enum import Enum
import httpx
import certifi

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

prediction_service = PredictionService()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameType(str, Enum):
    TEEN20 = "teen20"
    LUCKY7EU = "lucky7eu"
    DT20 = "dt20"

@app.on_event("startup")
async def startup_event():
    """Start verification loops for both games on application startup."""
    for game_type in GameType:
        prediction_service.start_verification_loop(game_type)
        logger.info(f"Started verification loop for {game_type}")

@app.get("/predict/{game_type}", response_model=PredictionResponse)
async def get_prediction(game_type: GameType):
    """Get predictions for a specific game type."""
    try:
        current_results = prediction_service.fetch_latest_data(game_type)
        predictions = prediction_service.predict_next_rounds(game_type)
        return PredictionResponse(
            current_results=current_results,
            predictions=predictions,
            game_type=game_type
        )
    except Exception as e:
        logger.error(f"Error getting prediction for {game_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/accuracy/{game_type}", response_model=ModelAccuracy)
async def get_model_accuracy(game_type: GameType):
    """Get accuracy metrics for a specific game type."""
    try:
        metrics = prediction_service.db.get_accuracy_metrics(game_type)
        return ModelAccuracy(
            timestamp=datetime.now().isoformat(),
            accuracy=metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0,
            total_predictions=metrics["total"],
            game_type=game_type
        )
    except Exception as e:
        logger.error(f"Error getting accuracy for {game_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/{game_type}")
async def get_model_performance(game_type: GameType):
    """Get detailed model performance metrics for a specific game type."""
    try:
        # Get recent accuracy metrics
        metrics = prediction_service.db.get_accuracy_metrics(game_type)
        
        # Get consecutive incorrect predictions
        consecutive_incorrect = prediction_service.db.get_consecutive_incorrect_predictions(game_type)
        
        # Get performance history
        performance_history = prediction_service.db.get_model_performance_history(game_type)
        
        return {
            "current_accuracy": metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0,
            "total_predictions": metrics["total"],
            "correct_predictions": metrics["correct"],
            "consecutive_incorrect": consecutive_incorrect,
            "performance_history": performance_history,
            "game_type": game_type,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics for {game_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/proxy/{game_type}")
async def proxy_request(game_type: str):
    """Proxy endpoint to handle API requests."""
    try:
        api_url = prediction_service.endpoints.get(game_type)
        if not api_url:
            raise HTTPException(status_code=400, detail=f"Invalid game type: {game_type}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Origin': 'https://terminal.apiserver.digital',
            'Referer': 'https://terminal.apiserver.digital/',
        }
        
        # Try with SSL verification first, fall back to unverified if needed
        try:
            async with httpx.AsyncClient(verify=certifi.where()) as client:
                response = await client.get(api_url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.SSLError:
            logger.warning(f"SSL verification failed for {game_type}, falling back to unverified connection")
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.get(api_url, headers=headers)
                response.raise_for_status()
                return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
