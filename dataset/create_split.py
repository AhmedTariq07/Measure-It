
from pathlib import Path
import random

IMAGE_DIR = Path("dataset/images")
OUTPUT_FILE = Path("dataset/split.txt")

SEED = 42

# Find all supported image files
images = sorted(
    path.name
    for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

if len(images) != 70:
    raise ValueError(
        f"Expected exactly 70 images, but found {len(images)}"
    )

# Shuffle reproducibly
random.Random(SEED).shuffle(images)

# Split into 70% train, 20% validation, 10% test
train = images[:49]
validation = images[49:63]
test = images[63:70]

# Save the fixed split
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("TRAIN:\n")
    f.write("\n".join(train))

    f.write("\n\nVALIDATION:\n")
    f.write("\n".join(validation))

    f.write("\n\nTEST:\n")
    f.write("\n".join(test))

print("Dataset split created successfully!")
print("Training images:", len(train))
print("Validation images:", len(validation))
print("Testing images:", len(test))
print("Saved to:", OUTPUT_FILE)
print("Random seed:", SEED)
