#!/usr/bin/env bash
# Memorus setup for OpenClaw
set -euo pipefail

MIN_PYTHON="3.9"
CONFIG_DIR="$HOME/.openclaw"
CONFIG_FILE="$CONFIG_DIR/memorus-config.json"

echo "=== Memorus × OpenClaw Setup ==="
echo ""

# Check Python version
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python >= $MIN_PYTHON"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    echo "ERROR: Python >= $MIN_PYTHON required, found $PY_VERSION"
    exit 1
fi

echo "[1/3] Python $PY_VERSION detected"

# Install memorus with MCP support
echo "[2/3] Installing memorus[mcp]..."
if pip install --only-binary :all: "memorus[mcp]" 2>/dev/null; then
    echo "  Installed (pre-built wheels)"
else
    echo "  Pre-built wheels unavailable, building from source..."
    pip install "memorus[mcp]"
fi

# Copy config template
echo "[3/3] Setting up configuration..."
mkdir -p "$CONFIG_DIR"
if [ -f "$CONFIG_FILE" ]; then
    echo "  Config already exists at $CONFIG_FILE (not overwriting)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    cp "$SCRIPT_DIR/memorus-config.example.json" "$CONFIG_FILE"
    echo "  Created $CONFIG_FILE"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Merge openclaw.json into your OpenClaw MCP config"
echo "  2. Edit $CONFIG_FILE if needed"
echo "  3. Add SKILL.md to your OpenClaw skills directory"
echo ""
echo "To migrate existing OpenClaw memories:"
echo "  python migrate.py --memory-dir ~/.openclaw"
