#!/bin/bash
# One command to run BERTrend on a RunPod pod (or any Linux GPU host).
#
#   cd /workspace/BERTrend && git pull && bash scripts/pod.sh
#
# Idempotent:
#   - first run  -> installs a self-contained venv in /workspace/.venv
#                   (Python 3.12 + bertrend + torch CUDA 12.8) and launches.
#   - next runs  -> skips the install, just (re)launches the two servers.
#
# Survives a pod Stop/Start IF the pod has a Volume disk mounted on /workspace
# (the venv and repo live there). Only the servers need restarting -> ~2 min.

set -e

WORKSPACE=/workspace
VENV="$WORKSPACE/.venv"
PY="$VENV/bin/python"
REPO="$WORKSPACE/BERTrend"

# --- Shared auth (must be identical across the 2 uvicorn workers) -----------
export BERTREND_SECRET_KEY=997c97c66d93821153c1a5c5fb7bab50a45cc87a6c7335b4d9225b0190ed4bd8
export BERTREND_CLIENT_SECRET=bd76aa472dd91aed4a56bf1935dbb802583c119824380d8567086579c0ef3324
export BERTREND_BASE_DIR="$REPO/.bertrend"
export EMBEDDING_SERVICE_URL=https://localhost:6464
export EMBEDDING_SERVICE_USE_LOCAL=false
mkdir -p "$BERTREND_BASE_DIR/logs"

# Secrets that must NOT live in the repo (OpenAI key for the LLM report).
# Create /workspace/.env on the pod once:
#   echo 'OPENAI_API_KEY=sk-...'        >  /workspace/.env
#   echo 'OPENAI_DEFAULT_MODEL=gpt-4o-mini' >> /workspace/.env
if [ -f "$WORKSPACE/.env" ]; then
    set -a
    . "$WORKSPACE/.env"
    set +a
    echo ">>> loaded $WORKSPACE/.env (OpenAI key for LLM analysis)"
else
    echo ">>> WARNING: no $WORKSPACE/.env -> LLM analysis will fail (no OpenAI key)"
fi

# --- uv (tiny binary, lives in ~/.local -> fast to (re)install) -------------
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
. "$HOME/.local/bin/env" 2>/dev/null || true

# --- One-time heavy install (skipped if the venv already exists) ------------
if [ ! -x "$PY" ]; then
    echo ">>> First run: building venv (Python 3.12 + bertrend + torch cu128)..."
    uv venv --python 3.12 --clear "$VENV"
    uv pip install --python "$PY" "$REPO"
    uv pip install --python "$PY" --force-reinstall torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128
    echo ">>> Install done."
else
    echo ">>> venv found, skipping install."
fi

# --- (Re)launch the two servers --------------------------------------------
pkill -f start.py    2>/dev/null || true
pkill -f streamlit   2>/dev/null || true
sleep 2

cd "$REPO/bertrend/services/embedding_server"
nohup "$PY" start.py > /tmp/emb.log 2>&1 &
echo ">>> embedding_server starting (PID $!)"

cd "$REPO/bertrend/demos/weak_signals"
nohup "$PY" -m streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=8084 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    > /tmp/st.log 2>&1 &
echo ">>> streamlit starting (PID $!)"

echo
echo "=== LANCÉ ==="
echo "Attends ~2 min le chargement du modèle :  tail -f /tmp/emb.log"
echo "Prêt quand tu vois 'Application startup complete' (2x)."
echo "Puis : page du Pod -> HTTP Service port 8084 -> upload ton .xlsx"
