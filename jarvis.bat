@echo off
rem jarvis.bat -- direct launcher for OpenJarvis.
rem Double-click = interactive chat. Or from terminal: jarvis.bat ask "question"
rem Starts Ollama automatically if it is not already running.

cd /d "%~dp0"
set "PATH=C:\Users\Dev\.local\bin;%PATH%"
set "JARVIS_NUM_CTX=8192"
set "OLLAMA_KEEP_ALIVE=2h"

rem Ensure Ollama is up (local model server, 127.0.0.1 only)
curl -s -o nul http://127.0.0.1:11434/api/version 2>nul
if errorlevel 1 (
    echo Starting Ollama...
    start "" /b "C:\Users\Dev\AppData\Local\Programs\Ollama\ollama.exe" serve
    timeout /t 3 /nobreak >nul
)

if "%~1"=="" (
    uv run jarvis chat
) else (
    uv run jarvis %*
)

rem Keep window open after chat ends when double-clicked
if "%~1"=="" pause
