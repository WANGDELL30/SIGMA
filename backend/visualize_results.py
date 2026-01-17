"""
Script untuk visualisasi grafik hasil training dan deteksi pada test images
"""
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import numpy as np
from PIL import Image

# ============================================
# 1. VISUALISASI GRAFIK TRAINING RESULTS
# ============================================
def plot_training_results():
    """Menampilkan grafik loss dan metrics dari training"""
    print("\n" + "=" * 50)
    print("VISUALISASI GRAFIK TRAINING")
    print("=" * 50)
    
    # Path ke results CSV
    csv_path = Path("runs/train/fruit_detection/results.csv")
    
    if not csv_path.exists():
        print(f"❌ File results.csv tidak ditemukan di: {csv_path}")
        print("Pastikan Anda sudah menjalankan train_model.py terlebih dahulu!")
        return
    
    # Baca CSV
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLOv8 Fruit Detection - Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    ax = axes[0, 0]
    loss_cols = [col for col in df.columns if 'loss' in col.lower()]
    for col in loss_cols:
        if col in df.columns:
            ax.plot(df.index, df[col], linewidth=2, label=col.replace('train/', '').replace('val/', ''))
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training & Validation Loss', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Precision & Recall
    ax = axes[0, 1]
    metric_cols = [col for col in df.columns if 'precision' in col.lower() or 'recall' in col.lower()]
    for col in metric_cols:
        if col in df.columns:
            ax.plot(df.index, df[col], linewidth=2, label=col.split('(')[0].strip())
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Precision & Recall', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: mAP Scores
    ax = axes[1, 0]
    map_cols = [col for col in df.columns if 'mAP' in col]
    for col in map_cols:
        if col in df.columns:
            ax.plot(df.index, df[col], linewidth=2, label=col, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('mAP', fontsize=11)
    ax.set_title('Mean Average Precision (mAP)', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate (jika ada)
    ax = axes[1, 1]
    lr_cols = [col for col in df.columns if 'lr' in col.lower()]
    if lr_cols:
        for col in lr_cols:
            if col in df.columns:
                ax.plot(df.index, df[col], linewidth=2, label=col)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Learning Rate Data', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    output_path = Path("results_training_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Grafik training disimpan ke: {output_path}")
    plt.show()


# ============================================
# 2. DETEKSI PADA TEST IMAGES
# ============================================
def detect_on_test_images():
    """Melakukan deteksi pada test images dan visualisasi hasilnya"""
    print("\n" + "=" * 50)
    print("SIMULASI DETEKSI PADA TEST IMAGES")
    print("=" * 50)
    
    # Load trained model
    model_path = Path("runs/train/fruit_detection/weights/best.pt")
    if not model_path.exists():
        print(f"❌ Model tidak ditemukan di: {model_path}")
        print("Pastikan Anda sudah menjalankan train_model.py terlebih dahulu!")
        return
    
    model = YOLO(str(model_path))
    print(f"✅ Model dimuat dari: {model_path}")
    
    # Tentukan direktori test images
    test_images_dir = Path("datasets/images/test")
    if not test_images_dir.exists():
        # Jika test tidak ada, gunakan semua images di datasets/images
        test_images_dir = Path("datasets/images")
    
    if not test_images_dir.exists():
        print(f"❌ Direktori images tidak ditemukan!")
        return
    
    # Cari semua image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(test_images_dir.rglob(f'*{ext}'))
        image_files.extend(test_images_dir.rglob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ Tidak ada image ditemukan di {test_images_dir}")
        return
    
    print(f"✅ Ditemukan {len(image_files)} images untuk deteksi")
    
    # Lakukan deteksi
    results_dir = Path("runs/detect/test_simulation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔍 Melakukan deteksi...")
    results = model.predict(
        source=str(test_images_dir),
        conf=0.25,
        iou=0.45,
        save=True,
        project='runs/detect',
        name='test_simulation',
        exist_ok=True,
        device=0,
    )
    
    print(f"✅ Deteksi selesai! Hasil disimpan di: {results_dir}")
    
    # Tampilkan beberapa hasil
    print("\n📊 RINGKASAN DETEKSI:")
    print("-" * 50)
    for i, result in enumerate(results[:5]):  # Tampilkan 5 hasil pertama
        if result.boxes:
            print(f"Image {i+1}: {len(result.boxes)} obyek terdeteksi")
            for box in result.boxes:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                print(f"  - {cls_name} (confidence: {conf:.2%})")
    
    return results_dir


# ============================================
# 3. VISUALISASI DETEKSI
# ============================================
def visualize_detections(results_dir):
    """Membuat gallery dari hasil deteksi"""
    print("\n" + "=" * 50)
    print("MEMBUAT GALLERY HASIL DETEKSI")
    print("=" * 50)
    
    # Cari semua image hasil deteksi
    detected_images = list(results_dir.glob('*.jpg')) + list(results_dir.glob('*.png'))
    
    if not detected_images:
        print(f"❌ Tidak ada hasil deteksi ditemukan")
        return
    
    detected_images = sorted(detected_images)[:9]  # Ambil 9 gambar pertama
    
    # Buat grid visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('YOLOv8 Fruit Detection Results', fontsize=18, fontweight='bold')
    
    for idx, (ax, img_path) in enumerate(zip(axes.flat, detected_images)):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"Detection {idx+1}", fontweight='bold')
        ax.axis('off')
    
    # Sembunyikan subplot kosong
    for idx in range(len(detected_images), 9):
        axes.flat[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path("results_detections_gallery.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Gallery disimpan ke: {output_path}")
    plt.show()


# ============================================
# 4. STATISTIK DETEKSI
# ============================================
def generate_detection_statistics(results_dir, model):
    """Generate statistik dari hasil deteksi"""
    print("\n" + "=" * 50)
    print("STATISTIK DETEKSI")
    print("=" * 50)
    
    image_files = list(results_dir.glob('*.jpg')) + list(results_dir.glob('*.png'))
    
    class_counts = {}
    confidence_scores = []
    
    for img_path in image_files:
        result = model.predict(str(img_path), conf=0.25, verbose=False)
        
        for box in result[0].boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            conf = box.conf[0].item()
            
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            confidence_scores.append(conf)
    
    # Visualisasi statistik
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Jumlah deteksi per class
    if class_counts:
        ax = axes[0]
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        ax.bar(classes, counts, color=colors[:len(classes)], edgecolor='black', linewidth=2)
        ax.set_ylabel('Jumlah Deteksi', fontsize=12, fontweight='bold')
        ax.set_title('Total Deteksi per Kelas', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Tambahkan nilai di atas bar
        for i, (cls, count) in enumerate(zip(classes, counts)):
            ax.text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Distribusi confidence scores
    if confidence_scores:
        ax = axes[1]
        ax.hist(confidence_scores, bins=20, color='#45B7D1', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.axvline(np.mean(confidence_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidence_scores):.3f}')
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribusi Confidence Scores', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path("results_detection_statistics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Statistik disimpan ke: {output_path}")
    plt.show()
    
    # Print summary
    print("\n📈 RINGKASAN STATISTIK:")
    print(f"Total deteksi: {sum(class_counts.values())}")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count} deteksi")
    if confidence_scores:
        print(f"Average confidence: {np.mean(confidence_scores):.3f}")


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("🚀 SISTEM VISUALISASI DAN SIMULASI DETEKSI BUAH")
    print("=" * 50)
    
    # 1. Tampilkan grafik training
    plot_training_results()
    
    # 2. Lakukan deteksi pada test images
    results_dir = detect_on_test_images()
    
    if results_dir:
        # 3. Visualisasi hasil deteksi
        visualize_detections(results_dir)
        
        # 4. Generate statistik
        model = YOLO(str(Path("runs/train/fruit_detection/weights/best.pt")))
        generate_detection_statistics(results_dir, model)
    
    print("\n" + "=" * 50)
    print("✅ SEMUA PROSES SELESAI!")
    print("=" * 50)
