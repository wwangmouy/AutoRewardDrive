#!/bin/bash
# AutoRewardDrive - No-Render Training Mode
# Both CARLA and training run without visualization

set -e

echo "=========================================="
echo "  AutoRewardDrive - No-Render Training"
echo "=========================================="
echo ""

# Save project directory
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check CARLA path
CARLA_ROOT=${CARLA_ROOT:-/home/wy/CARLA_0.9.13}
if [ ! -d "$CARLA_ROOT" ]; then
    echo "[WARNING] CARLA path not found: $CARLA_ROOT"
    echo "Please set environment variable: export CARLA_ROOT=/path/to/CARLA"
    read -p "Enter CARLA installation path: " CARLA_ROOT
fi

# Check if CARLA is already running
echo "[INFO] Checking CARLA server status"
if timeout 2 bash -c ">/dev/tcp/localhost/2000" 2>/dev/null; then
    echo "[INFO] CARLA server is already running"
else
    echo "[INFO] Starting CARLA server in offscreen mode"
    cd "$CARLA_ROOT"
    
    # Start CARLA in background with minimal GPU memory
    DISPLAY= ./CarlaUE4.sh \
        -quality-level=Low \
        -RenderOffScreen \
        -carla-rpc-port=2000 \
        -windowed \
        -ResX=320 \
        -ResY=240 \
        > /tmp/carla.log 2>&1 &
    
    CARLA_PID=$!
    echo "[INFO] CARLA started (PID: $CARLA_PID)"
    echo "[INFO] Waiting for CARLA to initialize"
    
    # Wait for CARLA to start (max 120 seconds)
    for i in {1..120}; do
        if timeout 2 bash -c ">/dev/tcp/localhost/2000" 2>/dev/null; then
            echo "[INFO] CARLA server ready"
            # Additional wait for full initialization
            echo "[INFO] Waiting for CARLA to fully initialize..."
            sleep 10
            break
        fi
        echo -n "."
        sleep 1
        if [ $i -eq 120 ]; then
            echo ""
            echo "[ERROR] CARLA startup timeout"
            echo "Check logs: tail -f /tmp/carla.log"
            exit 1
        fi
    done
fi

echo ""
echo "[INFO] Starting training"
echo "Press Ctrl+C to stop"
echo ""

# Return to project directory
cd "$PROJECT_DIR"

# Activate conda environment and start training
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vlm-rl

# Run training (no rendering)
python train.py \
    --host localhost \
    --port 2000 \
    --device cuda:0 \
    --total_timesteps 1000000 \
    --fps 15 \
    --log_dir tensorboard \
    --upper_update_freq 10 \
    --save_freq 10000

echo ""
echo "[INFO] Training completed"
