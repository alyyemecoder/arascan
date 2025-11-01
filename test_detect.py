from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\alish\Desktop\arascan\models\best.pt")

# Test with different confidence thresholds
for conf in [0.3, 0.2, 0.1, 0.05, 0.01]:
    print(f"\n=== Trying with confidence {conf} ===")
    results = model(r"C:\Users\alish\Downloads\harf ba.png", imgsz=1024, conf=conf, verbose=False)
    found = False

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            found = True
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                cls_name = r.names[cls_id]
                print(f"Detected: {cls_name}, Confidence: {conf_score:.2f}")

    if not found:
        print("No detections found.")
