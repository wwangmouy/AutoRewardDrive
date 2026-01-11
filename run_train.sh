#!/bin/bash
# AutoRewardDrive Training Script
# Usage: ./run_train.sh [mode]
# Modes: train, debug, auto, test

set -e

# Configuration
HOST="localhost"
PORT=2000
DEVICE="cuda:0"
TOTAL_TIMESTEPS=1000000
FPS=15
LOG_DIR="tensorboard"
UPPER_UPDATE_FREQ=10
SAVE_FREQ=10000

# Check if CARLA is running
check_carla() {
    echo "[INFO] Checking CARLA server on ${HOST}:${PORT}"
    if timeout 2 bash -c ">/dev/tcp/${HOST}/${PORT}" 2>/dev/null; then
        echo "[INFO] CARLA server is running"
        return 0
    else
        echo "[WARNING] CARLA server not detected on ${HOST}:${PORT}"
        return 1
    fi
}

# Mode 1: Standard Training (No Rendering)
train_standard() {
    echo "[INFO] Starting standard training (no rendering)"
    python train.py \
        --host ${HOST} \
        --port ${PORT} \
        --device ${DEVICE} \
        --total_timesteps ${TOTAL_TIMESTEPS} \
        --fps ${FPS} \
        --log_dir ${LOG_DIR} \
        --upper_update_freq ${UPPER_UPDATE_FREQ} \
        --save_freq ${SAVE_FREQ}
}

# Mode 2: Debug Training (With Rendering)
train_debug() {
    echo "[INFO] Starting debug training (with rendering)"
    python train.py \
        --host ${HOST} \
        --port ${PORT} \
        --device ${DEVICE} \
        --total_timesteps ${TOTAL_TIMESTEPS} \
        --fps ${FPS} \
        --log_dir ${LOG_DIR} \
        --upper_update_freq ${UPPER_UPDATE_FREQ} \
        --save_freq ${SAVE_FREQ} \
        --render
}

# Mode 3: Auto-start CARLA and Train
train_with_carla() {
    echo "[INFO] Starting CARLA server and training"
    python train.py \
        --host ${HOST} \
        --port ${PORT} \
        --device ${DEVICE} \
        --total_timesteps ${TOTAL_TIMESTEPS} \
        --fps ${FPS} \
        --log_dir ${LOG_DIR} \
        --upper_update_freq ${UPPER_UPDATE_FREQ} \
        --save_freq ${SAVE_FREQ} \
        --start_carla
}

# Mode 4: Quick Test (10k steps with rendering)
train_quick_test() {
    echo "[INFO] Starting quick test (10k steps with rendering)"
    python train.py \
        --host ${HOST} \
        --port ${PORT} \
        --device ${DEVICE} \
        --total_timesteps 10000 \
        --fps ${FPS} \
        --log_dir ${LOG_DIR}/test \
        --upper_update_freq ${UPPER_UPDATE_FREQ} \
        --save_freq 5000 \
        --render
}

# Main entry point
main() {
    echo "=========================================="
    echo "  AutoRewardDrive Training Launcher"
    echo "=========================================="
    echo ""

    MODE=${1:-train}

    case ${MODE} in
        train|standard)
            if ! check_carla; then
                echo "[ERROR] Please start CARLA server first"
                echo "[INFO] Run: cd /home/wy/CARLA_0.9.13 && ./CarlaUE4.sh -quality-level=Low -RenderOffScreen"
                exit 1
            fi
            train_standard
            ;;
        
        debug|render)
            if ! check_carla; then
                echo "[ERROR] Please start CARLA server first"
                exit 1
            fi
            train_debug
            ;;
        
        auto|carla)
            echo "[WARNING] This will attempt to start CARLA automatically"
            train_with_carla
            ;;
        
        test|quick)
            if ! check_carla; then
                echo "[ERROR] Please start CARLA server first"
                exit 1
            fi
            train_quick_test
            ;;
        
        help|--help|-h)
            echo "Usage: ./run_train.sh [mode]"
            echo ""
            echo "Available modes:"
            echo "  train    - Standard training (no rendering, fastest)"
            echo "  debug    - Debug training (with rendering)"
            echo "  auto     - Auto-start CARLA and train"
            echo "  test     - Quick test (10k steps with rendering)"
            echo "  help     - Show this help message"
            echo ""
            echo "Configuration:"
            echo "  HOST:              ${HOST}"
            echo "  PORT:              ${PORT}"
            echo "  DEVICE:            ${DEVICE}"
            echo "  TOTAL_TIMESTEPS:   ${TOTAL_TIMESTEPS}"
            echo "  FPS:               ${FPS}"
            echo "  LOG_DIR:           ${LOG_DIR}"
            echo "  UPPER_UPDATE_FREQ: ${UPPER_UPDATE_FREQ}"
            echo "  SAVE_FREQ:         ${SAVE_FREQ}"
            echo ""
            exit 0
            ;;
        
        *)
            echo "[ERROR] Unknown mode: ${MODE}"
            echo "Run './run_train.sh help' for usage information"
            exit 1
            ;;
    esac

    echo "[INFO] Training completed"
}

main "$@"
