@echo off
setlocal
title PhishGuard — Full Setup
color 0A
set PYTHONUTF8=1

echo ================================================================
echo   PHISHGUARD MULTI-MODAL AI — COMPLETE SETUP
echo   Capstone Project Setup + Model Training
echo ================================================================
echo.

echo [Step 1/3] Installing Python dependencies...
pip install flask flask-cors numpy scikit-learn joblib pandas scipy --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause & exit /b 1
)
echo   Dependencies installed OK.
echo.

echo [Step 2/3] Training ML Models...
echo   This will take 5-15 minutes on first run.
echo   URL Model  : 45,000 synthetic URLs + RF/GB/LR ensemble
echo   Text Model : 6,000 synthetic email texts + TF-IDF + LR
echo.
python train_models.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Training failed. Check the output above.
    pause & exit /b 1
)
echo.

echo [Step 3/3] Starting PhishGuard Server...
echo.
echo   Server   : http://127.0.0.1:5000
echo   Health   : http://127.0.0.1:5000/health
echo   Demo     : http://127.0.0.1:5000/demo
echo.
echo ================================================================
echo   EXTENSION SETUP:
echo   1. Open Chrome → chrome://extensions
echo   2. Enable Developer Mode (top-right toggle)
echo   3. Click "Load unpacked"
echo   4. Select folder: %~dp0extension
echo ================================================================
echo.
echo   Keep this window open during your demo!
echo.
python server.py
pause
