@echo off
REM Quick Start Script untuk Deteksi Buah YOLOv8
REM Jalankan script ini dari Command Prompt atau PowerShell

setlocal enabledelayedexpansion

echo.
echo =====================================
echo   SISTEM DETEKSI BUAH YOLOV8
echo =====================================
echo.

REM Cek Python tersedia
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python tidak ditemukan!
    echo Pastikan Python sudah diinstall dan ditambahkan ke PATH
    pause
    exit /b 1
)

echo Python terdeteksi: 
python --version
echo.

REM Menu utama
:menu
echo Pilih aksi:
echo.
echo 1. Install/Update Dependencies
echo 2. Training Model (100 epoch)
echo 3. Visualisasi Hasil Training
echo 4. Simulasi Deteksi (Image/Video/Webcam)
echo 5. Buka Folder Results
echo 6. Exit
echo.

set /p choice="Masukkan pilihan (1-6): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto train
if "%choice%"=="3" goto visualize
if "%choice%"=="4" goto simulate
if "%choice%"=="5" goto results
if "%choice%"=="6" goto end
if "%choice%"=="" goto menu

echo Pilihan tidak valid!
echo.
goto menu

:install
echo.
echo ===== INSTALL DEPENDENCIES =====
echo Installing required packages...
pip install ultralytics torch torchvision opencv-python matplotlib pandas pillow
echo.
echo Done! Tekan Enter untuk kembali ke menu.
pause
goto menu

:train
echo.
echo ===== TRAINING MODEL =====
echo Starting training... (ini bisa memakan waktu 30-60 menit)
echo.
python train_model.py
echo.
echo Done! Tekan Enter untuk kembali ke menu.
pause
goto menu

:visualize
echo.
echo ===== VISUALISASI HASIL =====
echo Generating visualization and statistics...
echo.
python visualize_results.py
echo.
echo Done! Tekan Enter untuk kembali ke menu.
pause
goto menu

:simulate
echo.
echo ===== SIMULASI DETEKSI =====
python simulate_detection.py
echo.
echo Done! Tekan Enter untuk kembali ke menu.
pause
goto menu

:results
echo.
echo Opening results folder...
start "" "runs"
echo.
goto menu

:end
echo.
echo Terima kasih! Sampai jumpa :)
echo.
pause
