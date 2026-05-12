#!/bin/bash
# Start Margaret's Companion agent — fused autonomy model + web UI
# Run: bash ~/start-companion.sh

set -e

PYTHON="$HOME/miniforge/bin/python3"
MODEL_PATH="$HOME/models/qwen3-30b-autonomy-fused"
MLX_PORT=8081
WEB_PORT=8080
WORKSPACE="$HOME/Projects"
SYSTEM_PROMPT="$HOME/lorepunk-system-prompt.md"

echo "=== Companion Agent ==="
echo "Model: Qwen3-30B-A3B + Autonomy LoRA (fused)"
echo "MLX server: port $MLX_PORT"
echo "Web UI: port $WEB_PORT"
echo ""

# Kill any existing instances
lsof -ti:$MLX_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$WEB_PORT | xargs kill -9 2>/dev/null || true
sleep 1

# Start MLX model server (OpenAI-compatible API)
echo "Starting model server..."
$PYTHON -m mlx_lm.server \
  --model "$MODEL_PATH" \
  --port $MLX_PORT \
  --host 127.0.0.1 &
MLX_PID=$!
echo "  MLX server PID: $MLX_PID"

# Wait for model server to be ready
echo "  Loading model into memory..."
until curl -s http://127.0.0.1:$MLX_PORT/v1/models > /dev/null 2>&1; do
  sleep 2
done
echo "  Model ready."

# Start web UI pointing at MLX server
echo "Starting web UI..."
cd $HOME/lorepunk
PYTHONPATH=$HOME/lorepunk $PYTHON web/app.py \
  --port $WEB_PORT \
  --model "qwen3-30b-autonomy" \
  --api-base "http://127.0.0.1:$MLX_PORT" \
  --api-type openai \
  --workspace "$WORKSPACE" \
  --system-prompt "$SYSTEM_PROMPT" &
WEB_PID=$!
echo "  Web UI PID: $WEB_PID"

sleep 2
echo ""
echo "=== Ready ==="
echo "Open http://localhost:$WEB_PORT in your browser"
echo "Model: Qwen3-30B-A3B with autonomy + honest uncertainty"
echo ""
echo "To stop: kill $MLX_PID $WEB_PID"

wait
