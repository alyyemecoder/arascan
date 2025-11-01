# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class DetectionResult(BaseModel):
    letter: str
    confidence: float
    bbox: Optional[List[int]] = None

class DetectResponse(BaseModel):
    detections: List[DetectionResult]

class LearnResponse(BaseModel):
    letter: str
    sound: Optional[str] = None
    example: Optional[str] = None
    related: Optional[List[dict]] = []

class PracticeIn(BaseModel):
    user_id: str
    letter: str
    score: float  # 0-100

class PracticeOut(BaseModel):
    score: float
    timestamp: str

class ProgressOut(BaseModel):
    letter: str
    score: float
    timestamp: str
