@echo off
echo Starting RIOM - Ambient Memory System...

echo.
echo 1. Starting Redis (Message Queue) via Docker...
docker start riom-redis || docker run -d --name riom-redis -p 6379:6379 redis
echo.

echo 2. Starting Capture Daemon...
start "Capture Daemon" cmd /k "cd /d C:\RIOM && C:\Users\User\anaconda3\python.exe -m capture.main"

echo 3. Starting AI Worker (OCR + LLM Processing)...
start "AI Worker" cmd /k "cd /d C:\RIOM && C:\Users\User\anaconda3\python.exe -m ai.worker"

echo 4. Starting API Server...
start "API Server" cmd /k "cd /d C:\RIOM && C:\Users\User\anaconda3\python.exe -m uvicorn api.main:app --port 8000"

echo.
echo All services have been started!
echo The API is available at: http://127.0.0.1:8000/docs
pause
