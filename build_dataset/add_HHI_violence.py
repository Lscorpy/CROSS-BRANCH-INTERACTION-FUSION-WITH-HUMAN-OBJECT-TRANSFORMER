import cv2
import json
import os

# ---------------------------
# Paths
# ---------------------------

IMAGE_DIR = "train_images"
ANNOTATION_DIR = "hoi_annotations.json"
OUTPUT_DIR = "build_data\test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "violence_branch_train.json")


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
verb_ID = {'hold': 0, 'stand': 1, 'sit': 2, 'ride': 3, 'walk': 4, 'look': 5, 
           'hit': 6, 'eat': 7, 'jump': 8, 'lay': 9, 'talk_on_phone': 10, 
           'carry': 11, 'throw': 12, 'catch': 13, 'cut': 14, 'run': 15, 
           'work_on_computer': 16, 'ski': 17, 'surf': 18, 'skateboard': 19, 'smile': 20, 
           'drink': 21, 'kick': 22, 'point': 23, 'read': 24, 'snowboard': 25, 
           'threaten': 26, 'raise': 27, 'aim': 28}


h_to_h_action=["threaten","attack","point_weapon_at", "destroy", "kidnapping"]
hh_action_ID={'threaten': 0, 'attack': 1, 'point_weapon_at': 2, 'destroy': 3, 'kidnapping': 4}

correct_label=["aim","hit","raise","hold","sit"]
vio_action_ID={'aim': 0, 'hit': 1, 'raise': 2, 'hold': 3, 'sit': 4}

obj_classes_originID={'N/A': 0, 'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 
 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 
 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 
 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 
 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 
'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 
'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 
'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 
'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 
'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 
'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 
'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 
'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 
'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 
'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 
'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 
'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 
'toothbrush': 90, 'gun': 91, 'stick': 92}

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

with open(ANNOTATION_DIR) as f:
    all_data = json.load(f)

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
        x1, y1, w, h = ann["bbox"]
        x2, y2 = x1 + w, y1 + h  # convert from xywh to xyxy
        label = f"{ann['id']}:{ann['actual_category']}"
        
        category = ann["actual_category"]
        
        color = (0,255,0) if category not in ["gun","stick", "baseball bat","knife","tennis racket"] else (0,0,255)

        if ann["id"] in selected:
            color = (255,0,0)  # BLUE for selected
        
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img_copy, label, (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if adding_bbox and bbox_start and bbox_end:
        cv2.rectangle(img_copy, bbox_start, bbox_end, (255,0,0), 2)

    cv2.putText(img_copy, "Click: Subject -> Object",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)
    
    cv2.putText(img_copy, "c: confirm | r: reset | a: add bbox | n: next | q: quit",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)
    cv2.putText(img_copy, "m: manual violence (ID input) | h: manual HOI (ID input)",
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)

    if adding_bbox:
        cv2.putText(img_copy, "Adding bbox: Click start, drag, release",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)

    return img_copy


def get_clicked_id(x, y, annotations):
    for ann in annotations:
        x1, y1, w, h = ann["bbox"]
        x2, y2 = x1 + w, y1 + h
        if x1 <= x <= x2 and y1 <= y <= y2:
            return ann["id"]
    return None

def print_verbs(h_to_h_action, per_line=5, col_width=22):
    print("\nChoose verb:")
    
    for i in range(0, len(h_to_h_action), per_line):
        row = []
        for j in range(i, min(i + per_line, len(h_to_h_action))):
            text = f"{j}:{h_to_h_action[j]}"
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

def choose_vio():
    while True:
        print_verbs(h_to_h_action)

        
        user_input = input("Enter violence verb index: ").strip()
        
        # check integer
        if not user_input.isdigit():
            print("❌ Invalid input. Please enter a number.")
            continue
        
        idx = int(user_input)
        
        # check range
        if idx < 0 or idx >= len(h_to_h_action):
            print("❌ Out of range. Try again.")
            continue
        
        return h_to_h_action[idx]

def save_vio_output(data, violence_annotations):
    global all_results, processed_files

    record = {
        "file_name": data["file_name"],
        "annotations": data["annotations"],
        "hoi_annotation": data["hoi_annotation"],
        "violence_annotation": violence_annotations.copy()  # save a copy of the current violence annotations
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

    # SAVE FULL DATASET
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
                w, h = x2 - x1, y2 - y1
                new_ann = {
                    "bbox": [x1, y1, w, h],  # store as xywh
                    "category_id": obj_classes_originID[category],
                    "actual_category": category,
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
                if annotations[obj_id]["actual_category"] != "person":
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
    if img_name in processed_files:
        current_idx += 1
        continue

    print(f"\n===== Image {current_idx+1}/{len(image_files)} =====")
    print(data["file_name"])
    print(data["hoi_annotation"])
    
    selected = []
    violence_annotations = []
    hoi_annotations = data["hoi_annotation"]
    
    cv2.namedWindow("Violence Tool")
    cv2.setMouseCallback("Violence Tool", mouse_callback, (annotations, data))
    
    while True:
        
        display = draw_boxes(image, annotations)
        cv2.imshow("Violence Tool", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            adding_bbox = True
            bbox_start = None
            bbox_end = None
            print("Adding mode: Click to start bbox, drag, release to end, then enter category.")
        
        # Confirm Violence
        if key == ord('h'):
            if len(selected) == 0:
                print("Select subject first")
                continue
            
            subject1 = selected[0]
            obj = selected[1] if len(selected) > 1 else -1
            
            verb = choose_verb()
            
            hoi_annotations.append({
                "subject_id": subject1,
                "object_id": obj,
                "category_id": verb_ID[verb],
                "verb_name": verb
            })
            
            print("Added:", hoi_annotations[-1])
            selected = []


        # Confirm Violence
        if key == ord('c'):
            if len(selected) == 0:
                print("Select subject first")
                continue
            
            subject = selected[0]
            obj = selected[1] if len(selected) > 1 else -1
            
            vio = choose_vio()
            
            
            violence_annotations.append({
                "subject_id": subject,
                "victim_id": obj,
                "target_visible": True if obj!=-1 else False,  # since we selected an object, it's visible
                "target_type": "human", # if obj != -1 and annotations[obj]["actual_category"] == "person" else "property",
                "category_id": hh_action_ID[vio],
                "violence_name": vio
            })
            
            print("Added:", violence_annotations[-1])
            selected = []

        if key == ord('m'):
            print("\n--- Manual Violence Mode ---")
            
            try:
                subject = int(input("Enter subject ID: "))
                vis = input("Is the target visible? (y/n): ").strip().lower() == 'y'
                if vis:
                    obj_input = input("Enter object ID (-1 if none): ").strip()
                    obj = int(obj_input) if obj_input != "" else -1
                display = draw_boxes(image, annotations)
                vio = choose_vio()

                target_type = input("Enter target type (property(1)/human(2)): ").strip()
                if target_type == "1":
                    target_type = "property"
                elif target_type == "2":
                    target_type = "human"
                else:
                    print("Invalid target type. Please enter 1 for property or 2 for human.")


                violence_annotations.append({
                    "subject_id": subject,
                    "victim_id": obj,
                    "target_visible": vis,
                    "target_type": target_type,
                    "category_id": hh_action_ID[vio],
                    "violence_name": vio
                })
                
                print("Added (manual):", violence_annotations[-1])
                selected = []  # reset click selection
                
            except Exception as e:
                print("❌ Invalid input, try again.")
        
        if key == ord('o'):
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
                    "category_id": verb_ID[verb],
                    "verb_name": verb
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
            save_vio_output(data, violence_annotations)
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