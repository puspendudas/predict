from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class Result(BaseModel):
    mid: str
    result: str

class GameState(BaseModel):
    status: str
    game_type: str
    current_mid: Optional[str] = None
    prediction: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    current_results: GameState
    game_type: str
    timestamp: str = datetime.now().isoformat()

class PredictionHistory(BaseModel):
    mid: str
    predicted_value: str
    actual_value: str
    timestamp: str
    was_correct: bool

class ModelAccuracy(BaseModel):
    accuracy: float
    total_predictions: int
    correct_predictions: int
    game_type: str
    timestamp: str = datetime.now().isoformat()

class DateRangeResults(BaseModel):
    game_type: str
    start_date: str
    end_date: str
    results: List[Dict[str, Any]]
    total_count: int
    timestamp: str = datetime.now().isoformat()
