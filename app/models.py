from pydantic import BaseModel
from typing import List

class Result(BaseModel):
    mid: str
    result: str

class PredictionResponse(BaseModel):
    current_results: List[Result]
    predictions: List[str]

class PredictionHistory(BaseModel):
    mid: str
    predicted_value: str
    actual_value: str
    timestamp: str
    was_correct: bool

class ModelAccuracy(BaseModel):
    timestamp: str
    accuracy: float
    total_predictions: int
