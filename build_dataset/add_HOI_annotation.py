import cv2
import json
import os

# ---------------------------
# Paths
# ---------------------------

IMAGE_DIR = "images"
ANNOTATION_DIR = "merged_annotations.json"
OUTPUT_DIR = "annotation\test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hoi_annotations.json")

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        all_results = json.load(f)
else:
    all_results = []

VCOCO_ACTIONS = [
    "hold", "stand", "sit", "ride", "walk", "look", "hit", "eat",
    "jump", "lay", "talk_on_phone", "carry", "throw", "catch",
    "cut", "run", "work_on_computer", "ski", "surf", "skateboard",
    "smile", "drink", "kick", "point", "read", "snowboard"
]

verbs = VCOCO_ACTIONS + ["threaten", "raise","aim"]

# Build Fast Lookup (CRITICAL)
processed_files = {item["file_name"]: item for item in all_results}

# ---------------------------
# Load image list
# ---------------------------
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

current_idx = 0

selected = []

adding_bbox = False
bbox_start = None
bbox_end = None

# ---------------------------
# Helper Functions
# ---------------------------
with open(ANNOTATION_DIR) as f:
    all_data = json.load(f)

# Fix bbox coordinates to ensure x1 < x2 and y1 < y2
for item in all_data:
    for ann in item["annotations"]:

        x1, y1, x2, y2 = ann["bbox"]
        
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        
        ann["bbox"] = [x1, y1, x2, y2]


def load_data(idx):
    img_name = image_files[idx]
    
    # find corresponding annotation
    data = next(d for d in all_data if d["file_name"] == img_name)
    
    image = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    
    for i, ann in enumerate(data["annotations"]):
        ann["id"] = i
    
    return image, data

def draw_boxes(img, annotations):
    img_copy = img.copy()
    
    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox"]
        label = f"{ann['id']}:{ann['category']}"
        
        category = ann["category"]
        
        # color = (0,255,0)  # GREEN for unselected
        color = (0,255,0) if category not in ["gun","stick", "baseball bat","knife","tennis racket"] else (0,0,255)

        if ann["id"] in selected:
            color = (255,0,0)  # BLUE for selected
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_copy, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if adding_bbox and bbox_start and bbox_end:
        cv2.rectangle(img_copy, bbox_start, bbox_end, (255,0,0), 2)

    # 🔥 ADD THIS UI TEXT
    cv2.putText(img_copy, "Click: Subject -> Object",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)
    
    # cv2.putText(img_copy, "c: confirm | r: reset | s: skip | n: next | q: quit",
    cv2.putText(img_copy, "c: confirm | r: reset | a: add bbox | n: next | q: quit",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)
    cv2.putText(img_copy, "m: manual select (ID input)",
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)

    if adding_bbox:
        cv2.putText(img_copy, "Adding bbox: Click start, drag, release",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)

    return img_copy

def get_clicked_id(x, y, annotations):
    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return ann["id"]
    return None

def print_verbs(verbs, per_line=5, col_width=22):
    print("\nChoose verb:")
    
    for i in range(0, len(verbs), per_line):
        row = []
        for j in range(i, min(i + per_line, len(verbs))):
            text = f"{j}:{verbs[j]}"
            row.append(f"{text:<{col_width}}")  # fixed width
        
        print("".join(row))

def choose_verb():
    while True:
        print_verbs(verbs)

        user_input = input("Enter verb index: ").strip()
        
        # check integer
        if not user_input.isdigit():
            print("❌ Invalid input. Please enter a number.")
            continue
        
        idx = int(user_input)
        
        # check range
        if idx < 0 or idx >= len(verbs):
            print("❌ Out of range. Try again.")
            continue
        
        return verbs[idx]


def save_output(data, hoi_annotations):
    global all_results, processed_files

    record = {
        "file_name": data["file_name"],
        "annotations": data["annotations"],
        "hoi_annotation": hoi_annotations.copy()
    }

    # If already exists → update
    if data["file_name"] in processed_files:
        for i, item in enumerate(all_results):
            if item["file_name"] == data["file_name"]:
                all_results[i] = record
                break
    else:
        all_results.append(record)

    # update lookup
    processed_files[data["file_name"]] = record

    # 🔥 SAVE FULL DATASET
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved dataset → {OUTPUT_FILE}")


# ---------------------------
# Mouse Callback
# ---------------------------
def mouse_callback(event, x, y, flags, param):
    global selected, adding_bbox, bbox_start, bbox_end
    
    annotations, data = param
    
    if adding_bbox:
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if bbox_start:
                bbox_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            bbox_end = (x, y)
            # Prompt for category
            category = input("Enter category: ").strip()
            if category:
                x1, y1 = bbox_start
                x2, y2 = bbox_end
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                new_ann = {
                    "category": category,
                    "bbox": [x1, y1, x2, y2],
                    "id": len(annotations)
                }
                annotations.append(new_ann)
                # Update all_data
                for item in all_data:
                    if item["file_name"] == data["file_name"]:
                        item["annotations"] = annotations
                        break
                # Save back to ANNOTATION_DIR
                with open(ANNOTATION_DIR, "w") as f:
                    json.dump(all_data, f, indent=2)
                print(f"Added new annotation: {new_ann}")
            adding_bbox = False
            bbox_start = None
            bbox_end = None
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            obj_id = get_clicked_id(x, y, annotations)
            if obj_id is None:
                return

            if len(selected) == 0:
                if annotations[obj_id]["category"] != "person":
                    print("First selection must be PERSON")
                    return

            if len(selected) >= 2:
                print("Already selected subject & object. Press 'c' or 'r'.")
                return

            selected.append(obj_id)
            if len(selected) == 1:
                print(f"Subject selected: {obj_id}")
            elif len(selected) == 2:
                print(f"Object selected: {obj_id}")

            print("Current selection:", selected)


# ---------------------------
# Main Loop
# ---------------------------
while current_idx < len(image_files):
    
    image, data = load_data(current_idx)
    annotations = data["annotations"]
    
    img_name = image_files[current_idx]

    # if img_name in processed_files:
    #     # print(f"Skipping already labeled: {img_name}")
    #     current_idx += 1
    #     continue
    print(f"\n===== Image {current_idx+1}/{len(image_files)} =====")
    print(data["file_name"])
    
    selected = []
    hoi_annotations = []
    
    cv2.namedWindow("HOI Tool")
    cv2.setMouseCallback("HOI Tool", mouse_callback, (annotations, data))
    
    while True:
        
        display = draw_boxes(image, annotations)
        cv2.imshow("HOI Tool", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            adding_bbox = True
            bbox_start = None
            bbox_end = None
            print("Adding mode: Click to start bbox, drag, release to end, then enter category.")
        
        # Confirm HOI
        if key == ord('c'):
            if len(selected) == 0:
                print("Select subject first")
                continue
            
            subject = selected[0]
            obj = selected[1] if len(selected) > 1 else -1
            
            verb = choose_verb()
            
            
            hoi_annotations.append({
                "subject_id": subject,
                "object_id": obj,
                "category_id": verb
            })
            
            print("Added:", hoi_annotations[-1])
            selected = []
        
        if key == ord('m'):
            print("\n--- Manual HOI Mode ---")
            
            try:
                subject = int(input("Enter subject ID: "))
                obj_input = input("Enter object ID (-1 if none): ").strip()
                obj = int(obj_input) if obj_input != "" else -1
                display = draw_boxes(image, annotations)
                verb = choose_verb()
                
                hoi_annotations.append({
                    "subject_id": subject,
                    "object_id": obj,
                    "category_id": verb
                })
                
                print("Added (manual):", hoi_annotations[-1])
                selected = []  # reset click selection
                
            except Exception as e:
                print("❌ Invalid input, try again.")

        # Reset selection
        if key == ord('r'):
            selected = []
            print("Reset selection")
        
        # Next image (SAVE + MOVE)
        if key == ord('n'):
            save_output(data, hoi_annotations)
            current_idx += 1
            cv2.destroyAllWindows()
            break
        
        # Previous image
        if key == ord('b') and current_idx > 0:
            current_idx -= 1
            cv2.destroyAllWindows()
            break
        
        # Quit
        if key == ord('q'):
            exit()

cv2.destroyAllWindows()
print("All images processed!") 