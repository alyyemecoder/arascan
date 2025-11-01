# app/routes/progress.py
from fastapi import APIRouter, HTTPException
from app.schemas import PracticeIn, PracticeOut, ProgressOut
from app.services.neo4j_service import create_practice_event, get_user_progress

router = APIRouter()

@router.post("/", response_model=PracticeOut)
def record_practice(payload: PracticeIn):
    practice = create_practice_event(payload.user_id, payload.letter, float(payload.score))
    if not practice:
        raise HTTPException(status_code=500, detail="Could not record practice.")
    return {"score": practice.get("score"), "timestamp": practice.get("timestamp").isoformat()}

@router.get("/{user_id}", response_model=list[ProgressOut])
def user_progress(user_id: str):
    records = get_user_progress(user_id)
    return records
