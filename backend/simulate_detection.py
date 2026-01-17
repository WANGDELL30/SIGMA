"""
Script untuk simulasi deteksi real-time dari video atau webcam
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
import sys

def draw_boxes(frame, results, model):
    """Menggambar bounding boxes pada frame"""
    if not results:
        return frame
    
    result = results[0]
    
    # Color palette
    colors = {
        'apel': (0, 0, 255),      # Red
        'pisang': (0, 255, 255)   # Yellow
    }
    
    for box in result.boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        
        # Get color
        color = colors.get(cls_name, (0, 255, 0))
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{cls_name} {conf:.2%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        label_ymin = max(y1, label_size[1] + 10)
        
        cv2.rectangle(frame, (x1, label_ymin - label_size[1] - 10), 
                     (x1 + label_size[0], label_ymin), color, -1)
        cv2.putText(frame, label, (x1, label_ymin - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def detect_from_video(video_path, model, output_path=None):
    """Deteksi dari file video"""
    print(f"\n🎬 Memproses video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Setup video writer jika ingin save
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    detections_count = 0
    
    print("🔍 Melakukan deteksi...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Resize untuk lebih cepat (optional)
        # frame = cv2.resize(frame, (640, 480))
        
        # Deteksi
        results = model.predict(frame, conf=0.25, verbose=False)
        
        # Count detections
        if results and len(results[0].boxes) > 0:
            detections_count += len(results[0].boxes)
        
        # Draw boxes
        frame = draw_boxes(frame, results, model)
        
        # Add info text
        info_text = f"Frame: {frame_count}/{total_frames} | Detections: {detections_count}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Show frame (jika ingin)
        # cv2.imshow('Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        # Save frame
        if output_path:
            out.write(frame)
        
        # Progress
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    if output_path:
        out.release()
    
    print(f"✅ Video processing selesai!")
    print(f"Total frames: {frame_count}, Total detections: {detections_count}")
    if output_path:
        print(f"Output disimpan ke: {output_path}")


def detect_from_images_sequence(images_dir, model, output_dir=None):
    """Deteksi dari sequence images dan buat video"""
    print(f"\n📸 Memproses image sequence dari: {images_dir}")
    
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print(f"❌ Direktori tidak ditemukan: {images_dir}")
        return
    
    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(sorted(images_dir.glob(ext)))
        image_files.extend(sorted(images_dir.glob(ext.upper())))
    
    if not image_files:
        print(f"❌ Tidak ada image ditemukan di {images_dir}")
        return
    
    print(f"✅ Ditemukan {len(image_files)} images")
    
    # Setup output
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    detections_total = 0
    detections_per_class = {}
    
    print("🔍 Melakukan deteksi...")
    
    for idx, img_path in enumerate(image_files):
        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        # Deteksi
        results = model.predict(frame, conf=0.25, verbose=False)
        
        # Count detections
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                detections_per_class[cls_name] = detections_per_class.get(cls_name, 0) + 1
                detections_total += 1
        
        # Draw boxes
        frame = draw_boxes(frame, results, model)
        
        # Add info
        cv2.putText(frame, f"Image: {idx+1}/{len(image_files)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save output
        if output_dir:
            output_path = output_dir / f"detected_{idx:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(image_files)} images...")
    
    print(f"✅ Image sequence processing selesai!")
    print(f"Total detections: {detections_total}")
    for cls_name, count in detections_per_class.items():
        print(f"  - {cls_name}: {count}")


def detect_from_webcam(model, duration=30):
    """Real-time detection dari webcam"""
    print(f"\n📹 Membuka webcam untuk {duration} detik...")
    print("Tekan 'q' untuk berhenti")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Tidak bisa membuka webcam!")
        return
    
    start_time = time.time()
    detections_count = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Deteksi
        results = model.predict(frame, conf=0.25, verbose=False)
        
        # Count detections
        if results and len(results[0].boxes) > 0:
            detections_count += len(results[0].boxes)
        
        # Draw boxes
        frame = draw_boxes(frame, results, model)
        
        # Add info
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        info = f"FPS: {fps:.1f} | Detections: {detections_count}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Webcam Detection', frame)
        
        # Check quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Check duration
        if time.time() - start_time > duration:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Webcam detection selesai!")
    print(f"Total frames: {frame_count}, Total detections: {detections_count}")
    print(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("🎬 SISTEM SIMULASI DETEKSI REAL-TIME")
    print("=" * 50)
    
    # Load model
    model_path = Path("runs/train/fruit_detection/weights/best.pt")
    if not model_path.exists():
        print(f"❌ Model tidak ditemukan!")
        sys.exit(1)
    
    model = YOLO(str(model_path))
    print(f"✅ Model dimuat: {model_path}\n")
    
    # Choose mode
    print("Pilih mode deteksi:")
    print("1. Image Sequence (folder dengan banyak gambar)")
    print("2. Single Video")
    print("3. Webcam Real-time")
    
    choice = input("\nMasukkan pilihan (1/2/3): ").strip()
    
    if choice == "1":
        images_dir = input("Masukkan path folder images (default: datasets/images): ").strip()
        if not images_dir:
            images_dir = "datasets/images"
        output_dir = input("Masukkan path folder output (tekan Enter untuk skip): ").strip()
        detect_from_images_sequence(images_dir, model, output_dir if output_dir else None)
    
    elif choice == "2":
        video_path = input("Masukkan path video file: ").strip()
        if not video_path:
            print("❌ Path tidak boleh kosong!")
        else:
            output_video = input("Masukkan path output video (tekan Enter untuk tidak save): ").strip()
            detect_from_video(video_path, model, output_video if output_video else None)
    
    elif choice == "3":
        duration = input("Berapa detik? (default: 30): ").strip()
        duration = int(duration) if duration.isdigit() else 30
        detect_from_webcam(model, duration)
    
    else:
        print("❌ Pilihan tidak valid!")
    
    print("\n" + "=" * 50)
    print("✅ Selesai!")
