@echo off
title PhishGuard AI Server
color 0A
echo ============================================================
echo   PHISHGUARD MULTI-MODAL AI - DEMO SERVER
echo ============================================================
echo.
echo [1/2] Installing required packages...
pip install flask flask-cors numpy --quiet
echo.
echo [2/2] Starting PhishGuard server...
echo.
echo Server:       http://127.0.0.1:5000
echo Health check: http://127.0.0.1:5000/health
echo Demo test:    http://127.0.0.1:5000/demo
echo.
echo ============================================================
echo  Keep this window open during your demo!
echo  Load the Chrome extension, then browse any website.
echo ============================================================
echo.
set PYTHONUTF8=1
python server.py
pause
