
import os
import json
import csv
import cv2
import numpy as np
import torch
from unet import UNet

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "undistorted_images")
MASK_DIR = os.path.join(BASE_DIR, "dataset", "masks", "test")
SPLIT_FILE = os.path.join(BASE_DIR, "dataset", "split.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unet.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "measurement")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 256
THRESHOLD = 0.5

# Load test filenames from split.txt
# Your split file uses headings such as TRAIN:, VALIDATION:, TEST:
test_files = []
reading_test = False

with open(SPLIT_FILE, "r") as f:
    for line in f:
        line = line.strip()

        if line == "TEST:":
            reading_test = True
            continue

        if line.endswith(":"):
            reading_test = False
            continue

        if reading_test and line:
            test_files.append(line)

if not test_files:
    raise RuntimeError("No TEST images found in dataset/split.txt")

print(f"Test images found: {len(test_files)}")

# Load trained model
model = UNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

results = []
total_tp = 0
total_fp = 0
total_fn = 0

def calculate_ap(ious, scores, threshold):
    """
    Approximate single-class AP.
    Assumes one ground-truth card and at most one prediction per image.
    """
    order = np.argsort(-np.asarray(scores))

    tp = 0
    fp = 0
    precision_at_tp = []

    for index in order:
        if ious[index] >= threshold:
            tp += 1
            precision_at_tp.append(tp / (tp + fp))
        else:
            fp += 1

    if len(ious) == 0:
        return 0.0

    return sum(precision_at_tp) / len(ious)


all_ious = []
all_scores = []

# Evaluate each test image
for filename in test_files:

    image_path = os.path.join(IMAGE_DIR, filename)

    mask_filename = os.path.splitext(filename)[0] + ".png"
    mask_path = os.path.join(MASK_DIR, mask_filename)

    image = cv2.imread(image_path)
    ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if ground_truth is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Resize image and mask
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    ground_truth = cv2.resize(
        ground_truth,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize image
    image = image.astype(np.float32) / 255.0

    # Convert to PyTorch tensor
    image = np.transpose(image, (2, 0, 1))

    image_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    # Model prediction
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(image_tensor)
        )[0, 0].cpu().numpy()

    # Binary masks
    predicted = probabilities >= THRESHOLD
    actual = ground_truth > 127

    # Pixel-level metrics
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())

    total_tp += tp
    total_fp += fp
    total_fn += fn

    # Mask IoU
    intersection = tp
    union = tp + fp + fn

    iou = intersection / union if union else 0.0

    # Prediction confidence
    score = (
        float(probabilities[predicted].mean())
        if predicted.any()
        else 0.0
    )

    all_ious.append(iou)
    all_scores.append(score)

    results.append({
        "filename": filename,
        "IoU": round(iou, 4),
        "confidence": round(score, 4),
        "TP_pixels": tp,
        "FP_pixels": fp,
        "FN_pixels": fn
    })

    print(f"Evaluated: {filename} | IoU: {iou:.4f}")

# Calculate final metrics
mean_iou = float(np.mean(all_ious))

pixel_precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp else 0.0
)

pixel_recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn else 0.0
)

# Approximate AP metrics
ap50 = calculate_ap(all_ious, all_scores, 0.50)

ap_values = [
    calculate_ap(all_ious, all_scores, threshold)
    for threshold in np.arange(0.50, 0.951, 0.05)
]

map_50_95 = float(np.mean(ap_values))

# Summary
summary = {
    "test_images": len(test_files),
    "mean_IoU": round(mean_iou, 4),
    "pixel_precision": round(pixel_precision, 4),
    "pixel_recall": round(pixel_recall, 4),
    "mAP_50": round(ap50, 4),
    "mAP_50_95": round(map_50_95, 4),
    "note": (
        "Single-class, one predicted card per image. "
        "AP is a mask-IoU-based approximation, "
        "not standard COCO mAP."
    )
}

# Save per-image results
csv_path = os.path.join(
    OUTPUT_DIR,
    "test_per_image.csv"
)

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

# Save summary JSON
json_path = os.path.join(
    OUTPUT_DIR,
    "test_metrics.json"
)

with open(json_path, "w") as f:
    json.dump(summary, f, indent=4)

# Display results
print("\nTEST EVALUATION COMPLETE")
print("------------------------")

for key, value in summary.items():
    print(f"{key}: {value}")

print(f"\nPer-image results saved: {csv_path}")
print(f"Summary saved: {json_path}")
