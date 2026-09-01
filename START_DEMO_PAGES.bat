@echo off
echo ========================================================
echo   PHISHGUARD MULTI-MODAL DEMO ENVIRONMENT
echo ========================================================
echo.
echo Starting a local Web Server JUST to host your frontend HTML demo pages...
echo (Your AI backend is still running on Render.com!)
echo Please ensure your PhishGuard Extension is enabled in Chrome/Edge.
echo The extension should point to your active server (local or cloud).
echo.
echo Press any key to open the Demo Hub in your default browser...
pause >nul

start http://127.0.0.1:8000/index.html

echo.
echo Frontend Demo Pages Hosted at: http://127.0.0.1:8000
echo Press Ctrl+C to close these demo pages when done.
echo ========================================================

python demo_server.py
