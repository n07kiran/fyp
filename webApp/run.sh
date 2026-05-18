#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# run.sh  –  Start both the FastAPI backend and Streamlit frontend
# Usage:  bash webApp/run.sh        (from any directory)
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

echo "🩸  AneRBC Anemia Classification System"
echo "========================================="

# ── resolve Python from project venv ─────────────────────────────
VENV_PYTHON="$REPO_ROOT/venv/bin/python"
VENV_STREAMLIT="$REPO_ROOT/venv/bin/streamlit"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌  Project venv not found at $REPO_ROOT/venv"
    echo "    Create it with:  python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

PYTHON="$VENV_PYTHON"
STREAMLIT="$VENV_STREAMLIT"

# Install webApp deps into the same venv if missing
if ! "$PYTHON" -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing webApp dependencies into project venv..."
    "$PYTHON" -m pip install -r requirements.txt
fi

# Kill any leftover processes on our ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

echo "🚀 Starting FastAPI backend on http://localhost:8000 ..."
"$PYTHON" backend.py &
BACKEND_PID=$!

sleep 4   # give backend time to load TF + model

echo "🌐 Starting Streamlit frontend on http://localhost:8501 ..."
"$STREAMLIT" run app.py --server.port 8501 --server.address localhost &
FRONTEND_PID=$!

echo ""
echo "✅ Both services started!"
echo "   Backend API docs : http://localhost:8000/docs"
echo "   Streamlit UI     : http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services."

cleanup() {
    echo "Stopping services…"
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

wait
