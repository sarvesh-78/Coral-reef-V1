import os
import random
from shutil import copyfile
from PIL import Image, ImageEnhance

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset_raw"          # folder containing Healthy, Bleached, Dead
OUTPUT_DIR = "dataset"               # folder to save train/val split
TARGET_COUNT = 720                   # number of images per class after balancing
VAL_SPLIT = 0.2                      # 20% of images go to validation
AUGMENTATIONS_PER_IMAGE = 2          # how many augmented images per original image
# ---------------------------------------

def augment_image(img):
    """Apply random augmentations to an image"""
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    # Random contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    return img

# Create train/val directories
for split in ["train", "val"]:
    for cls in os.listdir(DATASET_DIR):
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# Balance classes and split
for cls in os.listdir(DATASET_DIR):
    cls_path = os.path.join(DATASET_DIR, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    # Augment to reach TARGET_COUNT
    augmented_images = images.copy()
    while len(augmented_images) < TARGET_COUNT:
        img_name = random.choice(images)
        img = Image.open(os.path.join(cls_path, img_name))
        aug_img = augment_image(img)
        new_name = f"aug_{len(augmented_images)}_{img_name}"
        aug_img.save(os.path.join(cls_path, new_name))
        augmented_images.append(new_name)
    
    # Shuffle and split into train/val
    random.shuffle(augmented_images)
    split_idx = int(len(augmented_images) * (1 - VAL_SPLIT))
    train_images = augmented_images[:split_idx]
    val_images = augmented_images[split_idx:]
    
    # Copy to output folders
    for img_name in train_images:
        copyfile(os.path.join(cls_path, img_name), os.path.join(OUTPUT_DIR, "train", cls, img_name))
    for img_name in val_images:
        copyfile(os.path.join(cls_path, img_name), os.path.join(OUTPUT_DIR, "val", cls, img_name))

print("Dataset balancing and train/val split completed!")
