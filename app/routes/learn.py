# app/routes/learn.py
from fastapi import APIRouter, HTTPException
from app.services.neo4j_service import get_letter_node, get_similar_letters
from app.schemas import LearnResponse

router = APIRouter()

@router.get("/{letter}", response_model=LearnResponse)
def get_letter_info(letter: str):
    node = get_letter_node(letter)
    if not node:
        raise HTTPException(status_code=404, detail="Letter not found in knowledge graph.")
    related = get_similar_letters(letter)
    return {
        "letter": node.get("name"),
        "sound": node.get("sound"),
        "example": node.get("example"),
        "related": related
    }
