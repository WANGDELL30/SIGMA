"""
QUICK START - Panduan Singkat Menjalankan ML Detection
"""

print("""
╔══════════════════════════════════════════════════════════════════════╗
║            🍎 SISTEM DETEKSI BUAH DENGAN YOLOV8 🍌                 ║
║                                                                      ║
║  Proyekmu siap untuk:                                               ║
║  ✅ Training model deteksi buah (apel & pisang)                    ║
║  ✅ Generate grafik hasil training                                 ║
║  ✅ Simulasi deteksi real-time                                     ║
║  ✅ Analisis statistik deteksi                                     ║
╚══════════════════════════════════════════════════════════════════════╝

📋 FILE YANG TELAH DIBUAT:
─────────────────────────────────────────────────────────────────────

1. 🚂 train_model.py
   Melatih model YOLOv8 dengan dataset Anda
   Durasi: ~30-60 menit (dengan GPU)
   Jalankan dengan:
   $ python train_model.py

2. 📊 visualize_results.py
   Generate 3 grafik hasil training:
   - Training metrics (loss, precision, recall, mAP)
   - Gallery hasil deteksi pada test images
   - Statistik deteksi per kelas
   Jalankan dengan:
   $ python visualize_results.py

3. 🎬 simulate_detection.py
   Simulasi deteksi real-time dengan 3 mode:
   - Mode 1: Dari folder berisi banyak gambar
   - Mode 2: Dari file video
   - Mode 3: Dari webcam (interactive)
   Jalankan dengan:
   $ python simulate_detection.py

4. 📖 PANDUAN.md
   Panduan lengkap dengan tips & troubleshooting

5. 🎯 RUN.bat (Windows only)
   Script menu interaktif untuk Windows
   Double-click file ini untuk menjalankan


⚡ QUICK START (FASTEST WAY):
─────────────────────────────────────────────────────────────────────

Step 1: Setup (first time only)
--------
pip install ultralytics torch torchvision opencv-python matplotlib pandas pillow

Step 2: Train Model
--------
python train_model.py
(ini akan memakan waktu, pergi buat kopi dulu ☕)

Step 3: Lihat Hasil
--------
python visualize_results.py
(akan generate 3 grafik otomatis)

Step 4: Simulasi
--------
python simulate_detection.py
(pilih mode: 1=Images, 2=Video, 3=Webcam)


📁 OUTPUT FILES:
─────────────────────────────────────────────────────────────────────

Setelah menjalankan script, file output akan muncul di:

runs/train/fruit_detection/
├── weights/
│   ├── best.pt       ← Model terbaik (gunakan ini!)
│   └── last.pt
├── results.csv       ← Metrics training
└── events.out.tfevents...

results_training_metrics.png      ← Grafik training
results_detections_gallery.png    ← Gallery deteksi
results_detection_statistics.png  ← Statistik


🎯 WORKFLOW LENGKAP:
─────────────────────────────────────────────────────────────────────

┌─────────────────┐
│ INSTALL PACKAGE │ (pip install ...)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TRAINING MODEL  │ (train_model.py) - 30-60 min
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ VISUALISASI & ANALISIS  │ (visualize_results.py) - 2-5 min
│ - Grafik training       │
│ - Deteksi test images   │
│ - Statistik             │
└────────┬────────────────┘
         │
         ▼
┌──────────────────────┐
│ SIMULASI REAL-TIME   │ (simulate_detection.py)
│ - Images folder      │
│ - Video file         │
│ - Webcam live        │
└──────────────────────┘


💡 TIPS PENTING:
─────────────────────────────────────────────────────────────────────

✓ GPU vs CPU:
  - Dengan GPU (NVIDIA): ~40 menit training
  - Dengan CPU: ~3-4 jam training
  - Edit train_model.py baris "device=0" untuk ubah

✓ Jika Memory Error:
  - Di train_model.py, ubah batch=16 menjadi batch=8 atau 4

✓ Untuk Accuracy Lebih Baik:
  - Naikkan epochs di train_model.py dari 100 menjadi 200

✓ Model Size:
  - yolov8n.pt = Nano (~6 MB) - cepat, akurat sedang ✓ REKOMENDASI
  - yolov8s.pt = Small (~23 MB) - lebih akurat
  - yolov8m.pt = Medium (~49 MB) - paling akurat, lambat


📊 MEMAHAMI OUTPUT GRAFIK:
─────────────────────────────────────────────────────────────────────

Loss Graphs:
- TURUN = Model belajar ✓
- NAIK = Learning rate terlalu tinggi ✗
- DATAR = Sudah konvergen ✓

mAP Scores:
- Semakin tinggi semakin baik (0-1)
- Target: > 0.7 untuk deteksi yang baik

Confidence Scores:
- Rata-rata di atas 0.8 = Model percaya diri ✓
- Banyak di bawah 0.5 = Model kurang yakin ✗


🔧 TROUBLESHOOTING:
─────────────────────────────────────────────────────────────────────

❌ "CUDA out of memory"
✓ Kurangi batch size: batch=8 atau batch=4

❌ "data.yaml not found"
✓ Pastikan file ada di folder root dan jalankan script dari sana

❌ "Model tidak ada setelah training"
✓ Cek folder: runs/train/fruit_detection/weights/

❌ "results.csv not found"
✓ Training belum selesai atau error, cek log training


🚀 LANGKAH SELANJUTNYA SETELAH TRAINING:
─────────────────────────────────────────────────────────────────────

1. Evaluasi model accuracy dari grafik
2. Jika accuracy baik (mAP > 0.7):
   - Bisa deploy ke production
   - Export ke format ONNX atau TensorFlow
3. Jika accuracy kurang:
   - Tambah data training
   - Naikkan epochs
   - Fine-tune hyperparameters

Untuk fine-tuning lebih lanjut, lihat PANDUAN.md


📞 PERLU BANTUAN?
─────────────────────────────────────────────────────────────────────

1. Buka PANDUAN.md untuk penjelasan detail
2. Baca output/error messages di command prompt
3. Cek folder runs/ untuk log training lengkap
4. Baca dokumentasi: https://docs.ultralytics.com/


═══════════════════════════════════════════════════════════════════════

Sekarang Anda siap untuk:
✓ Melatih model YOLOv8 deteksi buah
✓ Generate visualisasi grafik training
✓ Simulasi deteksi dengan berbagai input
✓ Analisis performa model

HAPPY LEARNING! 🎉
Good luck dengan project machine learning Anda!

═══════════════════════════════════════════════════════════════════════
""")

# Interactive menu
if __name__ == "__main__":
    from pathlib import Path
    import subprocess
    import sys
    
    print("\nApa yang ingin Anda lakukan?")
    print("1. Install dependencies")
    print("2. Training model")
    print("3. Visualisasi hasil")
    print("4. Simulasi deteksi")
    print("5. Exit")
    print()
    
    choice = input("Pilih (1-5): ").strip()
    
    if choice == "1":
        print("\n📦 Installing packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "ultralytics", "torch", "torchvision", "opencv-python", 
                       "matplotlib", "pandas", "pillow"], check=True)
        print("✅ Done!")
    
    elif choice == "2":
        if not Path("train_model.py").exists():
            print("❌ train_model.py tidak ditemukan!")
        else:
            print("\n🚂 Starting training...")
            subprocess.run([sys.executable, "train_model.py"])
    
    elif choice == "3":
        if not Path("visualize_results.py").exists():
            print("❌ visualize_results.py tidak ditemukan!")
        else:
            print("\n📊 Generating visualizations...")
            subprocess.run([sys.executable, "visualize_results.py"])
    
    elif choice == "4":
        if not Path("simulate_detection.py").exists():
            print("❌ simulate_detection.py tidak ditemukan!")
        else:
            print("\n🎬 Starting simulation...")
            subprocess.run([sys.executable, "simulate_detection.py"])
    
    elif choice == "5":
        print("👋 Bye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice!")
