
import os
import sys
import json
import cv2
import numpy as np
import torch

# Import U-Net
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "models")
)
from unet import UNet

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unet.pth")
CALIBRATION_PATH = os.path.join(
    BASE_DIR, "calibration", "calibration_results.json"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "inference", "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 256
THRESHOLD = 0.5


def load_calibration():
    with open(CALIBRATION_PATH, "r") as f:
        data = json.load(f)

    camera_matrix = np.array(
        data["camera_matrix"], dtype=np.float64
    )

    distortion = np.array(
        data["distortion_coefficients"], dtype=np.float64
    )

    return camera_matrix, distortion


def load_model():
    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def predict_card(image_path):

    # Read original image
    original = cv2.imread(image_path)

    if original is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    height, width = original.shape[:2]

    # Load calibration
    camera_matrix, distortion = load_calibration()

    if (width, height) != (960, 1280):
        raise ValueError(
            f"Expected 960x1280 image, got {width}x{height}"
        )

    # Undistort using the same method as training dataset
    undistorted = cv2.undistort(
        original,
        camera_matrix,
        distortion,
        None,
        camera_matrix
    )

    # Prepare image for U-Net
    resized = cv2.resize(
        undistorted,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    rgb = rgb.astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))

    tensor = torch.from_numpy(
        rgb
    ).unsqueeze(0).to(DEVICE)

    # Run model
    model = load_model()

    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(tensor)
        )[0, 0].cpu().numpy()

    # Generate binary mask
    small_mask = (
        probabilities >= THRESHOLD
    ).astype(np.uint8)

    mask = cv2.resize(
        small_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("Card not detected.")
        return

    # Select largest contour
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(largest_contour)

    if area < 100:
        print("Detected region is too small.")
        return

    # Approximate card boundary as polygon
    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    epsilon = 0.02 * perimeter

    approx = cv2.approxPolyDP(
        largest_contour,
        epsilon,
        True
    )

    print(
        "Detected polygon corners:",
        len(approx)
    )

    # Create green segmentation overlay
    green = np.zeros_like(undistorted)
    green[:, :] = (0, 255, 0)

    overlay = np.where(
        mask[:, :, None] == 1,
        cv2.addWeighted(
            undistorted,
            0.5,
            green,
            0.5,
            0
        ),
        undistorted
    )

    # Draw approximated polygon boundary
    cv2.drawContours(
        overlay,
        [approx],
        -1,
        (0, 0, 255),
        3
    )

    # Mark polygon corners
    for point in approx:
        x, y = point[0]

        cv2.circle(
            overlay,
            (int(x), int(y)),
            7,
            (255, 0, 0),
            -1
        )

    # Save output images
    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    undistorted_path = os.path.join(
        OUTPUT_DIR,
        filename + "_undistorted.jpg"
    )

    overlay_path = os.path.join(
        OUTPUT_DIR,
        filename + "_prediction.jpg"
    )

    cv2.imwrite(
        undistorted_path,
        undistorted
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

    # Results
    print("\nPREDICTION COMPLETE")
    print("--------------------")
    print(f"Image dimensions: {width} x {height}")
    print(f"Predicted card area: {area:.0f} pixels")
    print(f"Polygon corners: {len(approx)}")
    print(f"Undistorted image: {undistorted_path}")
    print(f"Prediction overlay: {overlay_path}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python inference/predict.py <image_path>"
        )
        sys.exit(1)

    predict_card(sys.argv[1])
