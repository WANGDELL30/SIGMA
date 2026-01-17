"""
Script untuk melatih model YOLOv8 deteksi buah
"""
from ultralytics import YOLO
import torch

# Cek GPU availability
gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")
if gpu_available:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    device_to_use = 0
else:
    print("⚠️  GPU tidak ditemukan, menggunakan CPU")
    print("ℹ️  Training dengan CPU akan lebih lambat (~3-4 jam untuk 100 epoch)")
    device_to_use = 'cpu'

# Load model
model = YOLO('yolov8n.pt')

# Konfigurasi training
print("=" * 50)
print("MEMULAI TRAINING MODEL YOLOV8")
print("=" * 50)

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    device=device_to_use,  # Auto-detect GPU atau gunakan CPU
    patience=20,
    batch=4,  # Lebih kecil untuk CPU (dari 16)
    save=True,
    verbose=True,
    project='runs/train',
    name='fruit_detection',
    exist_ok=True,
    augment=True,
    cache='disk',
)

print("\n" + "=" * 50)
print("TRAINING SELESAI!")
print("=" * 50)
print(f"Model tersimpan di: runs/train/fruit_detection")
