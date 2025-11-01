# In app/routes/detect.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import shutil
import cv2
import numpy as np
from typing import List, Dict, Any
import logging
from ultralytics import YOLO

router = APIRouter(prefix="/detect", tags=["Detection"])

# Configure logging
logger = logging.getLogger(__name__)

# Directory to save uploaded images
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.options("/upload")
async def options_upload():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Validate file type
        allowed_extensions = {".jpg", ".jpeg", ".png"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Save uploaded file
        save_path = UPLOAD_DIR / file.filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Here you would typically call your detection logic
        # For now, we'll return a mock response
        response = {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "detections": []
        }
        
        return response

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{filename}")
async def get_uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)