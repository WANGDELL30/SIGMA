# 🍎🍌 SISTEM DETEKSI BUAH DENGAN YOLOV8

Panduan lengkap untuk melatih, visualisasi, dan simulasi model YOLOv8 deteksi buah (Apel dan Pisang).

## 📋 DAFTAR ISI
- [Instalasi](#instalasi)
- [Struktur File](#struktur-file)
- [Cara Penggunaan](#cara-penggunaan)
- [Penjelasan Script](#penjelasan-script)

---

## 🔧 INSTALASI

### 1. Install Requirements
```bash
pip install ultralytics torch torchvision opencv-python matplotlib pandas pillow
```

Jika menggunakan GPU (NVIDIA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verifikasi GPU (Optional)
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

## 📁 STRUKTUR FILE

```
machinelearningdeteksibuah/
├── data.yaml                    # Konfigurasi dataset
├── yolov8n.pt                   # Model pre-trained YOLOv8 Nano
├── datasets/
│   ├── images/
│   │   ├── train/               # Training images
│   │   ├── val/                 # Validation images
│   │   └── test/                # Test images
│   └── labels/                  # Annotation files (.txt)
├── train_model.py               # Script untuk training model
├── visualize_results.py          # Script untuk visualisasi & grafik
├── simulate_detection.py         # Script untuk simulasi deteksi real-time
├── plot_yolo_results.py         # Script visualisasi lama (optional)
├── runs/                        # Output dari training
│   ├── train/
│   │   └── fruit_detection/     # Hasil training
│   │       ├── weights/         # Model weights
│   │       │   ├── best.pt      # Best model
│   │       │   └── last.pt      # Last checkpoint
│   │       └── results.csv      # Training metrics
│   └── detect/
│       └── test_simulation/     # Hasil deteksi test images
├── results_*.png                # Output grafik dan visualisasi
└── README.md                    # File ini
```

---

## 🚀 CARA PENGGUNAAN

### STEP 1: Training Model
Training model YOLOv8 dengan dataset Anda:

```bash
python train_model.py
```

**Apa yang terjadi:**
- Model YOLOv8 Nano dimuat
- Training dilakukan selama 100 epoch
- Model terbaik disimpan di `runs/train/fruit_detection/weights/best.pt`
- Metrics disimpan di `runs/train/fruit_detection/results.csv`

**Durasi:** ~30-60 menit (dengan GPU), bisa lebih lama dengan CPU

**Catatan:** Baris `device=0` = GPU. Ganti dengan `device='cpu'` jika tidak ada GPU.

---

### STEP 2: Visualisasi Hasil Training & Simulasi Deteksi
Setelah training selesai, generate grafik dan deteksi test images:

```bash
python visualize_results.py
```

**Output yang dihasilkan:**
1. **results_training_metrics.png**
   - Loss Training & Validation
   - Precision & Recall
   - mAP Scores
   - Learning Rate

2. **results_detections_gallery.png**
   - Gallery 9 hasil deteksi terbaik
   - Bounding boxes dengan confidence score

3. **results_detection_statistics.png**
   - Bar chart jumlah deteksi per kelas
   - Histogram distribusi confidence scores

---

### STEP 3: Real-Time Simulation
Simulasi deteksi real-time dengan berbagai sumber input:

```bash
python simulate_detection.py
```

**Pilih mode:**
```
1. Image Sequence - Deteksi folder berisi banyak gambar
2. Single Video - Deteksi file video
3. Webcam Real-time - Deteksi langsung dari camera
```

**Contoh:**
```
Pilih mode deteksi (1/2/3): 1
Masukkan path folder images: datasets/images/test
Masukkan path folder output: runs/detect/sequence_output
```

---

## 📊 PENJELASAN SCRIPT

### 1. **train_model.py** - Training Model
```python
model.train(
    data='data.yaml',           # Lokasi konfigurasi dataset
    epochs=100,                 # Jumlah epoch training
    imgsz=640,                  # Input image size
    device=0,                   # GPU ID (0=GPU pertama, 'cpu'=CPU)
    patience=20,                # Early stopping jika tidak improve
    batch=16,                   # Batch size (kurangi jika out of memory)
    augment=True,              # Data augmentation
)
```

**Tips:**
- Jika memory error: kurangi `batch` ke 8 atau 4
- Untuk training lebih cepat: kurangi `epochs` menjadi 50
- Untuk accuracy lebih baik: naikkan `epochs` ke 200

---

### 2. **visualize_results.py** - Visualisasi & Analisis
Script ini melakukan 4 tahap:

#### A. Plot Training Results
Membaca `results.csv` dari training dan membuat 4 grafik:
- Loss curves (training & validation)
- Precision & Recall
- mAP metrics
- Learning rate

#### B. Detect on Test Images
Menggunakan best model untuk deteksi pada test images

#### C. Visualize Detections
Membuat gallery dari 9 hasil deteksi terbaik

#### D. Generate Statistics
Menghitung:
- Total deteksi per kelas (apel vs pisang)
- Distribusi confidence scores
- Average confidence

---

### 3. **simulate_detection.py** - Simulasi Real-Time

#### Mode 1: Image Sequence
```python
detect_from_images_sequence(
    images_dir="datasets/images/test",
    model=model,
    output_dir="runs/detect/sequence_output"
)
```
- Proses semua gambar di folder
- Simpan hasil dengan bounding boxes
- Print statistik deteksi

#### Mode 2: Single Video
```python
detect_from_video(
    video_path="video.mp4",
    model=model,
    output_path="output.mp4"
)
```
- Proses setiap frame video
- Tambahkan bounding boxes
- Bisa save ke video output

#### Mode 3: Webcam
```python
detect_from_webcam(
    model=model,
    duration=30  # 30 detik
)
```
- Real-time detection dari camera
- Display FPS & detection count
- Tekan 'q' untuk berhenti

---

## 📈 INTERPRETASI GRAFIK

### Loss Curves
- **Turun** = Model belajar dengan baik ✅
- **Naik** = Mungkin learning rate terlalu tinggi ❌
- **Flat** = Model sudah konvergen ✅

### Precision & Recall
- **Precision tinggi** = Sedikit false positives ✅
- **Recall tinggi** = Mendeteksi semua obyek ✅
- Keduanya tinggi = Model bagus! 🎉

### mAP Scores
- **mAP50** = Accuracy pada IoU 0.5
- **mAP50-95** = Average accuracy untuk IoU 0.5-0.95
- Semakin tinggi semakin baik

---

## 🎯 TIPS & TRICKS

### 1. Optimasi Training
```python
# Untuk training lebih cepat
epochs = 50
batch = 32  # Naikkan kalau punya GPU bagus

# Untuk accuracy lebih baik
epochs = 200
batch = 8   # Turunkan untuk update lebih sering
imgsz = 1280  # Naikkan ukuran input
```

### 2. Optimization untuk Inference
```python
# Gunakan FP16 precision (lebih cepat, sedikit kurang akurat)
model = YOLO('best.pt')
model.predict(source, half=True)  # FP16

# Atau export ke format lain yang lebih cepat
model.export(format='onnx')  # ONNX format
```

### 3. Deteksi dengan Confidence Threshold
```python
# Lebih ketat (kurangi false positives)
model.predict(source, conf=0.5)

# Lebih santai (tangkap lebih banyak deteksi)
model.predict(source, conf=0.25)
```

### 4. Debugging
```bash
# Cek GPU tersedia
python -c "import torch; print(torch.cuda.is_available())"

# Cek model size
python -c "from pathlib import Path; print(Path('runs/train/fruit_detection/weights/best.pt').stat().st_size / 1e6, 'MB')"

# Validasi data.yaml
python -c "import yaml; print(yaml.safe_load(open('data.yaml')))"
```

---

## ⚠️ TROUBLESHOOTING

### Error: CUDA out of memory
**Solusi:**
```python
# Di train_model.py, kurangi batch size
batch = 8  # atau 4
```

### Error: data.yaml not found
**Solusi:**
```bash
# Verifikasi file ada di root folder
ls -la data.yaml  # Linux/Mac
dir data.yaml     # Windows
```

### Model tidak ada setelah training
**Solusi:**
```bash
# Cek folder runs/train
ls -la runs/train/fruit_detection/weights/
```

### Webcam tidak terdeteksi
**Solusi:**
```python
# Di simulate_detection.py, coba device 1
cap = cv2.VideoCapture(1)  # Ganti 0 dengan 1 atau 2
```

---

## 📚 REFERENSI

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [PyTorch Docs](https://pytorch.org/docs/)

---

## 📝 NOTES

- Dataset harus memiliki struktur: `train/`, `val/`, `test/` subdirectories
- Image annotations harus dalam format YOLO (.txt files)
- Model akan otomatis menggunakan GPU jika tersedia
- Training memerlukan minimal 1-2 GB GPU memory

---

**Dibuat dengan ❤️ untuk deteksi buah yang akurat!**

Pertanyaan? Lihat logs di `runs/train/fruit_detection/` untuk detail lebih lanjut.
