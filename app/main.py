from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np
import os
from pathlib import Path
import shutil
import logging
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
BUILD_DIR = FRONTEND_DIR / "build"
UPLOAD_DIR = BASE_DIR / "uploads"

# Ensure upload directory exists
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# CORS Middleware Configuration - Only for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Serve static files from React build
if BUILD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(BUILD_DIR / "static")), name="static")

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    if request.method == "OPTIONS":
        response = JSONResponse(content={})
    else:
        response = await call_next(request)
    
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Arabic Letters Data
ARABIC_LETTERS = {
    "Alif": {"name": "Alif (ا)", "pronunciation": "A as in apple", "examples": ["أسد (Asad) – Lion", "أم (Umm) – Mother"]},
    "Ba": {"name": "Ba (ب)", "pronunciation": "B as in boy", "examples": ["باب (Baab) – Door", "بيت (Bayt) – House"]},
    "Ta": {"name": "Ta (ت)", "pronunciation": "T as in table", "examples": ["تفاح (Tuffah) – Apple", "تمساح (Timsah) – Crocodile"]},
    "Tha": {"name": "Tha (ث)", "pronunciation": "Th as in think", "examples": ["ثعلب (Tha‘lab) – Fox", "ثلج (Thalj) – Snow"]},
    "Jeem": {"name": "Jeem (ج)", "pronunciation": "J as in jam", "examples": ["جمل (Jamal) – Camel", "جبن (Jubn) – Cheese"]},
    "Hha": {"name": "Ḥā’ (ح)", "pronunciation": "Deep H, from throat", "examples": ["حب (Ḥubb) – Love", "حلم (Ḥulm) – Dream"]},
    "Kha": {"name": "Khā’ (خ)", "pronunciation": "Kh as in ‘Khalid’", "examples": ["خبز (Khubz) – Bread", "خروف (Kharūf) – Sheep"]},
    "Dal": {"name": "Dāl (د)", "pronunciation": "D as in dog", "examples": ["درج (Daraj) – Stairs", "دب (Dubb) – Bear"]},
    "Thal": {"name": "Thāl (ذ)", "pronunciation": "Th as in ‘this’", "examples": ["ذهب (Dhahab) – Gold", "ذراع (Dhira‘) – Arm"]},
    "Ra": {"name": "Rā’ (ر)", "pronunciation": "Rolled R", "examples": ["رجل (Rajul) – Man", "رأس (Ra’s) – Head"]},
    "Zay": {"name": "Zay (ز)", "pronunciation": "Z as in zebra", "examples": ["زهرة (Zahra) – Flower", "زيت (Zayt) – Oil"]},
    "Seen": {"name": "Sīn (س)", "pronunciation": "S as in sun", "examples": ["سمك (Samak) – Fish", "سماء (Samā’) – Sky"]},
    "Sheen": {"name": "Shīn (ش)", "pronunciation": "Sh as in shoe", "examples": ["شمس (Shams) – Sun", "شاي (Shay) – Tea"]},
    "Sad": {"name": "Ṣād (ص)", "pronunciation": "Heavy S", "examples": ["صبر (Ṣabr) – Patience", "صوت (Ṣawt) – Sound"]},
    "Dad": {"name": "Ḍād (ض)", "pronunciation": "Heavy D (unique to Arabic)", "examples": ["ضوء (Ḍaw’) – Light", "ضرب (Ḍaraba) – Hit"]},
    "Tta": {"name": "Ṭā’ (ط)", "pronunciation": "Emphatic T", "examples": ["طعام (Ṭa‘ām) – Food", "طريق (Ṭarīq) – Road"]},
    "Dha": {"name": "Ẓā’ (ظ)", "pronunciation": "Emphatic Th (like 'th' in 'though')", "examples": ["ظرف (Ẓarf) – Envelope", "ظلام (Ẓalām) – Darkness"]},
    "Ain": {"name": "‘Ayn (ع)", "pronunciation": "Deep 'A' from throat", "examples": ["عين ('Ayn) – Eye", "عمل ('Amal) – Work"]},
    "Ghain": {"name": "Ghayn (غ)", "pronunciation": "Gh (as in French 'r')", "examples": ["غرفة (Ghurfa) – Room", "غنم (Ghanam) – Sheep"]},
    "Fa": {"name": "Fā' (ف)", "pronunciation": "F as in fish", "examples": ["فم (Fam) – Mouth", "فيل (Fīl) – Elephant"]},
    "Qaf": {"name": "Qāf (ق)", "pronunciation": "Deep K", "examples": ["قلب (Qalb) – Heart", "قمر (Qamar) – Moon"]},
    "Kaf": {"name": "Kāf (ك)", "pronunciation": "K as in kite", "examples": ["كتاب (Kitāb) – Book", "كرسي (Kursī) – Chair"]},
    "Lam": {"name": "Lām (ل)", "pronunciation": "L as in lamp", "examples": ["لبن (Laban) – Milk", "لعب (La'b) – Play"]},
    "Meem": {"name": "Mīm (م)", "pronunciation": "M as in moon", "examples": ["مدرسة (Madrasa) – School", "ماء (Mā') – Water"]},
    "Noon": {"name": "Nūn (ن)", "pronunciation": "N as in nose", "examples": ["نار (Nār) – Fire", "نجم (Najm) – Star"]},
    "Ha": {"name": "Hā' (هـ)", "pronunciation": "Soft H (as in hello)", "examples": ["هواء (Hawā') – Air", "هاتف (Hātif) – Phone"]},
    "Waw": {"name": "Wāw (و)", "pronunciation": "W as in water", "examples": ["وردة (Wardah) – Flower", "ولد (Walad) – Boy"]},
    "Ya": {"name": "Yā' (ي)", "pronunciation": "Y as in yes", "examples": ["يد (Yad) – Hand", "يوم (Yawm) – Day"]}
}

# Initialize YOLO model
MODEL_PATH = Path(r"c:\Users\alish\Desktop\arascan\models\best.pt")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

try:
    model = YOLO(str(MODEL_PATH))
    logger.info("YOLO model loaded successfully")
    # Print model info for debugging
    logger.info(f"Model classes: {model.names if hasattr(model, 'names') else 'No classes found'}")
except Exception as e:
    logger.error(f"Error loading YOLO model: {str(e)}")
    raise

def process_detection_results(results):
    """Process YOLO detection results into a structured format."""
    # Map model class names to ARABIC_LETTERS keys
    CLASS_MAPPING = {
        "Ain": "Ain", "Alif": "Alif", "Ba": "Ba", "Dal": "Dal", "Dhod": "Dad",
        "Dzal": "Dha", "Dzo": "Tha", "Fa": "Fa", "Ghain": "Ghain", "Haa": "Ha",
        "Hamzah": "Alif", "Jim": "Jeem", "Kaf": "Kaf", "Kha": "Kha", "Kho": "Hha",
        "Lam-": "Lam", "LamAlif": "Lam", "Mim": "Meem", "Nun": "Noon", "Qaf": "Qaf",
        "Ro": "Ra", "Shod": "Sad", "Sin": "Seen", "Syin": "Sheen", "Ta": "Ta",
        "Tho": "Tha", "Tsa": "Tta", "Waw": "Waw", "Yaa": "Ya", "Za": "Zay"
    }

    detections = []
    letters_info = {}
    
    if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
        logger.warning("No detections found in the results")
        return {"detections": [], "letters": {}}
    
    result = results[0]
    
    if len(result.boxes) == 0:
        logger.warning("No bounding boxes found in the results")
        return {"detections": [], "letters": {}}
    
    for box in result.boxes:
        try:
            conf = float(box.conf[0].cpu().numpy())
            if conf < 0.3:
                continue
                
            cls = int(box.cls[0].cpu().numpy())
            cls_name = result.names[cls]
            
            # Map the detected class to our standard naming
            letter_key = CLASS_MAPPING.get(cls_name, cls_name)
            
            logger.info(f"Detected {cls_name} (mapped to {letter_key}) with confidence {conf:.2f}")
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            
            detections.append({
                "letter": letter_key,
                "original_class": cls_name,  # Keep original for reference
                "confidence": conf,
                "bbox": {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            })
            
            if letter_key not in letters_info:
                letters_info[letter_key] = ARABIC_LETTERS.get(letter_key, {
                    "name": f"{letter_key} (Detected as {cls_name})",
                    "pronunciation": "Not available",
                    "examples": []
                })
                
        except Exception as e:
            logger.error(f"Error processing detection: {str(e)}")
            continue
    
    logger.info(f"Processed {len(detections)} detections")
    return {
        "detections": detections,
        "letters": letters_info
    }

class DetectionResponse(BaseModel):
    message: str
    original_image: str
    annotated_image: str
    detections: List[Dict[str, Any]]
    letters: Dict[str, Dict[str, Any]]
    detection_count: int

# Create uploads directory if it doesn't exist
# Upload directory is now defined at the top with other paths

@app.post("/detect/upload", response_model=DetectionResponse)
async def upload_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file upload: {file.filename}")
        
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

        # Create uploads directory if it doesn't exist
        UPLOAD_DIR.mkdir(exist_ok=True)
        
        # Save uploaded file
        save_path = UPLOAD_DIR / file.filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file to: {save_path}")

        # Read and process image using vision_service
        from app.services.vision_service import detect_letters, ARABIC_LETTERS
        
        # Log the ARABIC_LETTERS keys for debugging
        logger.info(f"Available ARABIC_LETTERS keys: {list(ARABIC_LETTERS.keys())}")
        
        # Perform detection
        detection_results = detect_letters(str(save_path))
        
        # Extract detections and letters info
        detections = detection_results.get("detections", [])
        letters_info = detection_results.get("letters", {})
        
        logger.info(f"Processed {len(detections)} detections")
        logger.info(f"Detected letters: {[d['letter'] for d in detections]}")
        logger.info(f"Letters info keys: {list(letters_info.keys())}")
        
        # Create annotated image
        img = cv2.imread(str(save_path))
        if img is None:
            raise HTTPException(status_code=500, detail="Failed to read the uploaded image")
            
        annotated_img = img.copy()
        
        # Draw bounding boxes on the image
        for det in detections:
            try:
                bbox = det["bbox"]
                cv2.rectangle(
                    annotated_img,
                    (int(bbox["x1"]), int(bbox["y1"])),
                    (int(bbox["x2"]), int(bbox["y2"])),
                    (0, 255, 0),  # Green color
                    2  # Thickness
                )
                
                # Add label with confidence
                label = f"{det['letter']} {det['confidence']:.2f}"
                cv2.putText(
                    annotated_img,
                    label,
                    (int(bbox["x1"]), int(bbox["y1"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Font scale
                    (0, 255, 0),  # Green color
                    1  # Thickness
                )
            except Exception as e:
                logger.error(f"Error drawing box for detection {det}: {str(e)}")
                continue
        
        # Save annotated image
        output_filename = f"annotated_{file.filename}"
        output_path = UPLOAD_DIR / output_filename
        cv2.imwrite(str(output_path), annotated_img)
        logger.info(f"Saved annotated image to: {output_path}")

        # Prepare learning output - use the letters_info directly as it's already processed in vision_service
        learning_output = {}
        for letter_key, letter_info in letters_info.items():
            if letter_key in ARABIC_LETTERS:
                learning_output[letter_key] = {
                    "name": ARABIC_LETTERS[letter_key]["name"],
                    "pronunciation": ARABIC_LETTERS[letter_key]["pronunciation"],
                    "examples": ARABIC_LETTERS[letter_key]["examples"]
                }
            else:
                logger.warning(f"Letter {letter_key} not found in ARABIC_LETTERS despite being in letters_info")
        
        logger.info(f"Learning output prepared for {len(learning_output)} letters: {list(learning_output.keys())}")
        
        # Log the final response structure for debugging
        logger.info(f"Sample learning output for first letter: {next(iter(learning_output.values()), 'No letters detected')}")

        # Prepare response with proper structure
        response = {
            "message": "Detection completed successfully",
            "original_image": f"/uploads/{file.filename}",
            "annotated_image": f"/uploads/{output_filename}",
            "detections": detections,
            "letters": learning_output,  # This now has the complete learning data
            "detection_count": len(detections)
        }
        
        # Log the response structure (without the image data)
        debug_response = response.copy()
        debug_response["detections"] = [{"letter": d["letter"], "confidence": d["confidence"]} for d in debug_response["detections"]]
        logger.info(f"API Response: {debug_response}")

        return response

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Debug endpoint to check model information and class mappings
@app.get("/model/classes")
async def get_model_classes():
    """Return the raw class names from the model and their mappings."""
    if not hasattr(model, 'names'):
        return {"error": "Model not properly loaded or doesn't have class names"}
    
    CLASS_MAPPING = {
        "Ain": "Ain", "Alif": "Alif", "Ba": "Ba", "Dal": "Dal", "Dhod": "Dad",
        "Dzal": "Dha", "Dzo": "Tha", "Fa": "Fa", "Ghain": "Ghain", "Haa": "Ha",
        "Hamzah": "Alif", "Jim": "Jeem", "Kaf": "Kaf", "Kha": "Kha", "Kho": "Hha",
        "Lam-": "Lam", "LamAlif": "Lam", "Mim": "Meem", "Nun": "Noon", "Qaf": "Qaf",
        "Ro": "Ra", "Shod": "Sad", "Sin": "Seen", "Syin": "Sheen", "Ta": "Ta",
        "Tho": "Tha", "Tsa": "Tta", "Waw": "Waw", "Yaa": "Ya", "Za": "Zay"
    }
    
    # Create a mapping of model class indices to their mapped names
    class_mappings = {}
    for idx, name in model.names.items():
        class_mappings[int(idx)] = {
            "original_name": name,
            "mapped_to": CLASS_MAPPING.get(name, name),
            "in_arabic_letters": CLASS_MAPPING.get(name, name) in ARABIC_LETTERS
        }
    
    return {
        "model_type": str(type(model)),
        "num_classes": len(model.names),
        "class_mappings": class_mappings,
        "available_arabic_letters": list(ARABIC_LETTERS.keys())
    }

# Add endpoint to serve uploaded files
@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# Serve React frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = BUILD_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>Arabic Character Recognition</h1>
                    <p>Frontend build files not found. Please build the frontend first.</p>
                </body>
            </html>
            """,
            status_code=404
        )
    return FileResponse(index_path)

# API root endpoint
@app.get("/api")
async def api_root():
    return {"message": "Welcome to the Arabic Character Recognition API"}

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}