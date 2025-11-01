import os
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from ultralytics import YOLO

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("vision_service.log"), logging.StreamHandler()],
)
logger = logging.getLogger("VisionService")

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best.pt"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Model path: {MODEL_PATH}, exists: {MODEL_PATH.exists()}")
logger.info(f"Upload directory: {UPLOAD_DIR}")

# ----------------------------
# Thresholds
# ----------------------------
CONFIDENCE_THRESHOLD = 0.3

# ----------------------------
# Arabic Letters Dictionary
# ----------------------------
ARABIC_LETTERS = {
    "Alif": {"name": "Alif (ا)", "pronunciation": "A as in apple", "examples": ["أسد (Asad) – Lion", "أم (Umm) – Mother"]},
    "Ba": {"name": "Ba (ب)", "pronunciation": "B as in boy", "examples": ["باب (Baab) – Door", "بيت (Bayt) – House"]},
    "Ta": {"name": "Ta (ت)", "pronunciation": "T as in table", "examples": ["تفاح (Tuffah) – Apple", "تمساح (Timsah) – Crocodile"]},
    "Tha": {"name": "Tha (ث)", "pronunciation": "Th as in think", "examples": ["ثعلب (Tha'lab) – Fox", "ثلج (Thalj) – Snow"]},
    "Jeem": {"name": "Jeem (ج)", "pronunciation": "J as in jam", "examples": ["جمل (Jamal) – Camel", "جبن (Jubn) – Cheese"]},
    "Hha": {"name": "Ḥā' (ح)", "pronunciation": "Deep H, from throat", "examples": ["حب (Ḥubb) – Love", "حلم (Ḥulm) – Dream"]},
    "Kha": {"name": "Khā' (خ)", "pronunciation": "Kh as in 'Khalid'", "examples": ["خبز (Khubz) – Bread", "خروف (Kharūf) – Sheep"]},
    "Dal": {"name": "Dāl (د)", "pronunciation": "D as in dog", "examples": ["درج (Daraj) – Stairs", "دب (Dubb) – Bear"]},
    "Thal": {"name": "Thāl (ذ)", "pronunciation": "Th as in 'this'", "examples": ["ذهب (Dhahab) – Gold", "ذراع (Dhira') – Arm"]},
    "Ra": {"name": "Rā' (ر)", "pronunciation": "Rolled R", "examples": ["رجل (Rajul) – Man", "رأس (Ra's) – Head"]},
    "Zay": {"name": "Zay (ز)", "pronunciation": "Z as in zebra", "examples": ["زهرة (Zahra) – Flower", "زيت (Zayt) – Oil"]},
    "Seen": {"name": "Sīn (س)", "pronunciation": "S as in sun", "examples": ["سمك (Samak) – Fish", "سماء (Samā') – Sky"]},
    "Sheen": {"name": "Shīn (ش)", "pronunciation": "Sh as in shoe", "examples": ["شمس (Shams) – Sun", "شاي (Shay) – Tea"]},
    "Sad": {"name": "Ṣād (ص)", "pronunciation": "Heavy S", "examples": ["صبر (Ṣabr) – Patience", "صوت (Ṣawt) – Sound"]},
    "Dad": {"name": "Ḍād (ض)", "pronunciation": "Heavy D", "examples": ["ضوء (Ḍaw') – Light", "ضرب (Ḍaraba) – Hit"]},
    "Tta": {"name": "Ṭā' (ط)", "pronunciation": "Emphatic T", "examples": ["طعام (Ṭa'ām) – Food", "طريق (Ṭarīq) – Road"]},
    "Dha": {"name": "Ẓā' (ظ)", "pronunciation": "Emphatic Th", "examples": ["ظرف (Ẓarf) – Envelope", "ظلام (Ẓalām) – Darkness"]},
    "Ain": {"name": "'Ayn (ع)", "pronunciation": "Deep A", "examples": ["عين ('Ayn) – Eye", "عمل ('Amal) – Work"]},
    "Ghain": {"name": "Ghayn (غ)", "pronunciation": "Gh (as in French 'r')", "examples": ["غرفة (Ghurfa) – Room", "غنم (Ghanam) – Sheep"]},
    "Fa": {"name": "Fā' (ف)", "pronunciation": "F as in fish", "examples": ["فم (Fam) – Mouth", "فيل (Fīl) – Elephant"]},
    "Qaf": {"name": "Qāf (ق)", "pronunciation": "Deep K", "examples": ["قلب (Qalb) – Heart", "قمر (Qamar) – Moon"]},
    "Kaf": {"name": "Kāf (ك)", "pronunciation": "K as in kite", "examples": ["كتاب (Kitāb) – Book", "كرسي (Kursī) – Chair"]},
    "Lam": {"name": "Lām (ل)", "pronunciation": "L as in lamp", "examples": ["لبن (Laban) – Milk", "لعب (La'b) – Play"]},
    "Meem": {"name": "Mīm (م)", "pronunciation": "M as in moon", "examples": ["مدرسة (Madrasa) – School", "ماء (Mā') – Water"]},
    "Noon": {"name": "Nūn (ن)", "pronunciation": "N as in nose", "examples": ["نار (Nār) – Fire", "نجم (Najm) – Star"]},
    "Ha": {"name": "Hā' (هـ)", "pronunciation": "Soft H", "examples": ["هواء (Hawā') – Air", "هاتف (Hātif) – Phone"]},
    "Waw": {"name": "Wāw (و)", "pronunciation": "W as in water", "examples": ["وردة (Wardah) – Flower", "ولد (Walad) – Boy"]},
    "Ya": {"name": "Yā' (ي)", "pronunciation": "Y as in yes", "examples": ["يد (Yad) – Hand", "يوم (Yawm) – Day"]},
}

# ----------------------------
# Load YOLO Model
# ----------------------------
_model = None

def load_model():
    """Load YOLO model once."""
    global _model
    if _model is None:
        logger.info(f"Loading YOLO model from {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
        logger.info("YOLO model loaded successfully")
        logger.info(f"Model classes: {_model.names if hasattr(_model, 'names') else 'None'}")
    return _model

# ----------------------------
# Process Detection Results
# ----------------------------
def process_detection_results(results):
    """Map YOLO results to Arabic learning data."""
    # Expanded class mapping to handle all possible model outputs
    CLASS_MAPPING = {
        # Direct mappings
        'Ain': 'Ain', 'Alif': 'Alif', 'Alef': 'Alif', 'Ba': 'Ba', 'Beh': 'Ba',
        'Ta': 'Ta', 'Teh': 'Ta', 'Tha': 'Tha', 'Theh': 'Tha', 'Jeem': 'Jeem',
        'Hha': 'Hha', 'Hah': 'Hha', 'Kha': 'Kha', 'Khah': 'Kha', 'Dal': 'Dal',
        'Thal': 'Thal', 'Ra': 'Ra', 'Zay': 'Zay', 'Za': 'Zay', 'Seen': 'Seen',
        'Sheen': 'Sheen', 'Shin': 'Sheen', 'Sad': 'Sad', 'Dad': 'Dad', 'Dhad': 'Dad',
        'Tta': 'Tta', 'Tah': 'Tta', 'Dha': 'Dha', 'Ain': 'Ain', 'Ayn': 'Ain',
        'Ghain': 'Ghain', 'Ghayn': 'Ghain', 'Fa': 'Fa', 'Qaf': 'Qaf', 'Kaf': 'Kaf',
        'Lam': 'Lam', 'Meem': 'Meem', 'Noon': 'Noon', 'Ha': 'Ha', 'Waw': 'Waw',
        'Ya': 'Ya', 'Yaa': 'Ya', 'Hamza': 'Alif', 'Hamzah': 'Alif'
    }

    detections = []
    letters_info = {}

    if not results:
        logger.warning("No results returned by YOLO.")
        return {"detections": [], "letters": {}}
        
    # Get the first result (batch size is 1)
    result = results[0] if isinstance(results, list) else results
    
    # Check if we have any detections
    if not hasattr(result, 'boxes') or result.boxes is None:
        logger.warning("No bounding boxes found in results.")
        return {"detections": [], "letters": {}}
        
    logger.info(f"YOLO detected {len(result.boxes)} potential letters")

    # Process each detection
    for box in result.boxes:
        try:
            # Get confidence score
            conf = float(box.conf[0].cpu().numpy())
            if conf < CONFIDENCE_THRESHOLD:
                logger.debug(f"Skipping detection with low confidence: {conf:.2f}")
                continue
                
            # Get class ID and name
            cls_id = int(box.cls[0].cpu().numpy())
            original = result.names[cls_id]
            
            # Map to our standard letter name
            mapped = CLASS_MAPPING.get(original, original)
            
            # Try case-insensitive match if needed
            if mapped not in ARABIC_LETTERS:
                # Try case-insensitive match
                matched = False
                for letter in ARABIC_LETTERS.keys():
                    if letter.lower() == mapped.lower():
                        mapped = letter
                        matched = True
                        break
                
                if not matched:
                    logger.warning(f"Could not map detected class '{original}' to a known Arabic letter")
                    continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            
            # Add to detections
            detections.append({
                "letter": mapped,
                "original_class": original,
                "confidence": conf,
                "bbox": {
                    "x1": x1, 
                    "y1": y1, 
                    "x2": x2, 
                    "y2": y2,
                    "width": x2 - x1, 
                    "height": y2 - y1
                }
            })
            
            # Add to learning info if not already present
            if mapped not in letters_info:
                letters_info[mapped] = ARABIC_LETTERS[mapped]
                logger.info(f"Added learning info for letter: {mapped}")
        except Exception as e:
            logger.error(f"Error processing detection: {e}")

    logger.info(f"Final detections: {len(detections)} letters recognized.")
    return {"detections": detections, "letters": letters_info}

# ----------------------------
# Preprocess Image
# ----------------------------
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

# ----------------------------
# Main Detection
# ----------------------------
def detect_letters(image_path: str):
    """Detect Arabic letters in an image and return learning information."""
    start_time = time.time()
    logger.info(f"Starting letter detection for: {image_path}")
    
    try:
        # Validate image path
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return {"detections": [], "letters": {}}

        # Read the image
        original_img = cv2.imread(image_path)
        if original_img is None:
            logger.error(f"Failed to read image: {image_path}")
            return {"detections": [], "letters": {}}
            
        logger.info(f"Image loaded successfully. Dimensions: {original_img.shape}")

        # Preprocess the image
        processed = preprocess_image(original_img)
        debug_path = image_path.replace(".", "_preprocessed.")
        cv2.imwrite(debug_path, processed)
        logger.info(f"Saved preprocessed image to: {debug_path}")
        
        # Load the model
        model = load_model()
        logger.info("Model loaded successfully")
        
        # Try different confidence thresholds if needed
        confidence_thresholds = [0.3, 0.25, 0.2, 0.15, 0.1]
        best_result = {"detections": [], "letters": {}}
        
        for conf in confidence_thresholds:
            logger.info(f"Trying detection with confidence threshold: {conf}")
            
            # Run inference
            results = model(original_img, conf=conf, imgsz=1024, verbose=False)
            
            # Process results
            current_result = process_detection_results(results)
            
            # If we found more detections than before, update best result
            if len(current_result["detections"]) > len(best_result["detections"]):
                best_result = current_result
                logger.info(f"New best result with {len(best_result['detections'])} detections at conf={conf}")
                
                # If we found all possible letters, no need to try lower thresholds
                if len(best_result["detections"]) >= 5:  # Arbitrary number, adjust based on expected letters
                    break
        
        # Log final results
        detection_time = time.time() - start_time
        logger.info(f"Detection completed in {detection_time:.2f}s")
        logger.info(f"Letters recognized: {list(best_result['letters'].keys())}")
        
        # Ensure we have the full learning data for each detected letter
        final_letters = {}
        for letter in best_result['letters']:
            if letter in ARABIC_LETTERS:
                final_letters[letter] = ARABIC_LETTERS[letter]
            else:
                logger.warning(f"Letter '{letter}' not found in ARABIC_LETTERS")
        
        return {
            "detections": best_result["detections"],
            "letters": final_letters
        }

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return {"detections": [], "letters": {}}
