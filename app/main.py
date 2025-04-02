from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.predict import PredictionService
from app.models import PredictionResponse, ModelAccuracy, DateRangeResults
import asyncio
import logging
from datetime import datetime
from enum import Enum
import httpx
import certifi
from app.config.logging_config import setup_logging

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize logging
logger = setup_logging()

prediction_service = PredictionService()

class GameType(str, Enum):
    TEEN20 = "teen20"
    LUCKY7EU = "lucky7eu"
    DT20 = "dt20"

@app.on_event("startup")
async def startup_event():
    """Start verification loops for all game types on application startup."""
    game_types = ["teen20", "lucky7eu", "dt20"]
    for game_type in game_types:
        try:
            prediction_service.start_verification_loop(game_type)
            logger.info(f"Started verification loop for {game_type}")
        except Exception as e:
            logger.error(f"Failed to start verification loop for {game_type}: {str(e)}")

@app.get("/predict/{game_type}", response_model=PredictionResponse)
async def get_prediction(game_type: GameType):
    """Get predictions for a specific game type."""
    try:
        # Get current game state
        game_state = prediction_service.get_current_game_state(game_type)
        
        if game_state["status"] == "error":
            raise HTTPException(status_code=404, detail=game_state["message"])
        
        return PredictionResponse(
            current_results=game_state,
            game_type=game_type,
            timestamp=datetime.now().isoformat()
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
            correct_predictions=metrics["correct"],
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
            "current_accuracy": metrics["accuracy"],
            "total_predictions": metrics["total"],
            "correct_predictions": metrics["correct"],
            "incorrect_predictions": metrics["incorrect"],
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

@app.get("/results/{game_type}", response_model=DateRangeResults)
async def get_results_by_date_range(
    game_type: GameType,
    start_date: str,
    end_date: str
):
    """Get results for a specific game type within a date range."""
    try:
        results = prediction_service.db.get_results_by_date_range(game_type, start_date, end_date)
        return DateRangeResults(
            game_type=game_type,
            start_date=start_date,
            end_date=end_date,
            results=results,
            total_count=len(results)
        )
    except Exception as e:
        logger.error(f"Error getting results for {game_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
