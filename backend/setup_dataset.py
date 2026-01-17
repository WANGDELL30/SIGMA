"""
Script untuk split dataset menjadi train/val/test folders
"""
import os
import shutil
from pathlib import Path
import random

print("=" * 60)
print("MENYIAPKAN DATASET (Split Train/Val/Test)")
print("=" * 60)

# Setup paths
base_dir = Path("datasets")
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# Check if images and labels exist
if not images_dir.exists():
    print(f"❌ Folder {images_dir} tidak ditemukan!")
    exit(1)

if not labels_dir.exists():
    print(f"❌ Folder {labels_dir} tidak ditemukan!")
    exit(1)

# Get all image files
image_files = []
for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
    image_files.extend(sorted(images_dir.glob(f'*{ext}')))

print(f"✅ Ditemukan {len(image_files)} image files")

if len(image_files) == 0:
    print("❌ Tidak ada image ditemukan!")
    exit(1)

# Split ratio: 70% train, 15% val, 15% test
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Shuffle
random.seed(42)  # For reproducibility
random.shuffle(image_files)

# Calculate split indices
train_count = int(len(image_files) * train_ratio)
val_count = int(len(image_files) * val_ratio)
# test_count = len(image_files) - train_count - val_count

train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

print(f"\n📊 Split Dataset:")
print(f"  Train: {len(train_files)} images ({train_ratio*100:.0f}%)")
print(f"  Val:   {len(val_files)} images ({val_ratio*100:.0f}%)")
print(f"  Test:  {len(test_files)} images ({test_ratio*100:.0f}%)")

# Create directory structure
splits = {
    'train': train_files,
    'valid': val_files,
    'test': test_files
}

for split_name, files in splits.items():
    # Create directories
    split_images_dir = base_dir / split_name / 'images'
    split_labels_dir = base_dir / split_name / 'labels'
    
    split_images_dir.mkdir(parents=True, exist_ok=True)
    split_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Processing {split_name}...")
    
    for img_file in files:
        # Image copy
        dst_img = split_images_dir / img_file.name
        shutil.copy2(img_file, dst_img)
        
        # Find and copy corresponding label file
        label_name = img_file.stem + '.txt'
        src_label = labels_dir / label_name
        
        if src_label.exists():
            dst_label = split_labels_dir / label_name
            shutil.copy2(src_label, dst_label)
    
    print(f"  ✅ {len(files)} images + labels copied to {split_name}/")

print("\n" + "=" * 60)
print("✅ DATASET SIAP!")
print("=" * 60)
print("\nStuktur folder sekarang:")
print("datasets/")
print("├── train/")
print("│   ├── images/ (image files)")
print("│   └── labels/ (annotation files)")
print("├── valid/")
print("│   ├── images/")
print("│   └── labels/")
print("└── test/")
print("    ├── images/")
print("    └── labels/")
