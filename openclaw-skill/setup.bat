@echo off
REM Memorus setup for OpenClaw (Windows)
setlocal enabledelayedexpansion

set "MIN_PYTHON=3.9"
set "CONFIG_DIR=%USERPROFILE%\.openclaw"
set "CONFIG_FILE=%CONFIG_DIR%\memorus-config.json"

echo === Memorus x OpenClaw Setup ===
echo.

REM Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found. Please install Python ^>= %MIN_PYTHON%
    exit /b 1
)

for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PY_VERSION=%%v"
for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

if !PY_MAJOR! LSS 3 (
    echo ERROR: Python ^>= %MIN_PYTHON% required, found %PY_VERSION%
    exit /b 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 9 (
    echo ERROR: Python ^>= %MIN_PYTHON% required, found %PY_VERSION%
    exit /b 1
)

echo [1/3] Python %PY_VERSION% detected

REM Install memorus with MCP support
echo [2/3] Installing memorus[mcp]...
pip install --only-binary :all: "memorus[mcp]" >nul 2>&1
if errorlevel 1 (
    echo   Pre-built wheels unavailable, building from source...
    pip install "memorus[mcp]"
) else (
    echo   Installed (pre-built wheels^)
)

REM Copy config template
echo [3/3] Setting up configuration...
if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"
if exist "%CONFIG_FILE%" (
    echo   Config already exists at %CONFIG_FILE% (not overwriting^)
) else (
    copy "%~dp0memorus-config.example.json" "%CONFIG_FILE%" >nul
    echo   Created %CONFIG_FILE%
)

echo.
echo === Setup Complete ===
echo.
echo Next steps:
echo   1. Merge openclaw.json into your OpenClaw MCP config
echo   2. Edit %CONFIG_FILE% if needed
echo   3. Add SKILL.md to your OpenClaw skills directory
echo.
echo To migrate existing OpenClaw memories:
echo   python migrate.py --memory-dir %USERPROFILE%\.openclaw
