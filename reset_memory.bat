@echo off
echo ==========================================
echo    RIOM - COMPLETE MEMORY WIPE
echo ==========================================
echo.
echo WARNING: This will permanently delete ALL recorded screenshots, 
echo AI summaries, and your entire search history. 
echo.
echo Please make sure you have closed all three RIOM terminal windows
echo before proceeding.
echo.
pause

echo.
echo 1. Deleting SQLite Database...
if exist "C:\RIOM\ambient_memory.db" del /f /q "C:\RIOM\ambient_memory.db"

echo 2. Deleting ChromaDB (Vector Search History)...
if exist "C:\RIOM\chroma_db" rmdir /s /q "C:\RIOM\chroma_db"

echo 3. Deleting All Saved Screenshots...
if exist "C:\RIOM\data\frames" rmdir /s /q "C:\RIOM\data\frames"

echo.
echo All memory has been completely erased!
echo You now have a 100%% clean slate. 
echo Run start_all.bat to begin fresh.
echo.
pause
