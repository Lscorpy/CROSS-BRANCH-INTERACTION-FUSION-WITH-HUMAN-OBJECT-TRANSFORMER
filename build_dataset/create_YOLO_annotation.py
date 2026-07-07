from ultralytics import YOLO
import os
import json

# =========================
# CONFIG
# =========================
image_folder = r"D:\coding\FYP_Project\utils\train_extracted_frames\sabotage_violence"
output_json  = r"D:\coding\FYP_Project\utils\train_extracted_frames\sabotage_yolo_annotations.json"
conf_thresh  = 0.3

model = YOLO("yolov8s.pt")  # s= COCO classes=80
names = model.names

image_exts = (".jpg", ".jpeg", ".png")
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith(image_exts)
]
all_results = []

# =========================
# RUN DETECTION
# =========================
for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)

    result = model(img_path, conf=conf_thresh, device=0)[0]

    annotations = []

    if result.boxes is not None:
        for box, cls, conf in zip(
            result.boxes.xyxy,
            result.boxes.cls,
            result.boxes.conf
        ):
            if conf < conf_thresh:
                continue

            class_id = int(cls)
            class_name = names[class_id]

            x1, y1, x2, y2 = map(int, box.tolist())

            annotations.append({
                "category": class_name,
                "bbox": [x1, y1, x2, y2]
            })

    all_results.append({
        "annotations": annotations,
        "file_name": img_name
    })

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print(f"✅ Detection complete. Results saved to:\n{output_json}")