#!/usr/bin/env python3
"""
AutoRewardDrive Training Launcher (Python Version)
Usage: python run_train.py [mode]
"""

import os
import sys
import socket
import subprocess
import argparse
from pathlib import Path

# ==================== Configuration ====================
CONFIG = {
    'host': 'localhost',
    'port': 2000,
    'device': 'cuda:0',
    'total_timesteps': 1_000_000,
    'fps': 15,
    'log_dir': 'tensorboard',
    'upper_update_freq': 10,
    'save_freq': 10000,
}

# ==================== Color Output ====================
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def print_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

# ==================== Utility Functions ====================
def check_carla(host='localhost', port=2000, timeout=2):
    """Check if CARLA server is running"""
    print_info(f"Checking if CARLA server is running on {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            print_success("CARLA server is running")
            return True
        else:
            print_warning(f"CARLA server not detected on {host}:{port}")
            return False
    except Exception as e:
        print_warning(f"Could not check CARLA server: {e}")
        return False

def build_train_command(config, render=False, start_carla=False):
    """Build training command"""
    cmd = [
        sys.executable, 'train.py',
        '--host', config['host'],
        '--port', str(config['port']),
        '--device', config['device'],
        '--total_timesteps', str(config['total_timesteps']),
        '--fps', str(config['fps']),
        '--log_dir', config['log_dir'],
        '--upper_update_freq', str(config['upper_update_freq']),
        '--save_freq', str(config['save_freq']),
    ]
    
    if render:
        cmd.append('--render')
    if start_carla:
        cmd.append('--start_carla')
    
    return cmd

def run_training(cmd):
    """Run training with subprocess"""
    try:
        print_info(f"Running: {' '.join(cmd)}")
        print("=" * 60)
        process = subprocess.run(cmd)
        return process.returncode == 0
    except KeyboardInterrupt:
        print_warning("\nTraining interrupted by user")
        return False
    except Exception as e:
        print_error(f"Training failed: {e}")
        return False

# ==================== Training Modes ====================
def train_standard(config):
    """Standard training (no rendering for speed)"""
    print_info("Starting STANDARD training (no rendering for speed)...")
    cmd = build_train_command(config, render=False)
    return run_training(cmd)

def train_debug(config):
    """Debug training (with rendering)"""
    print_info("Starting DEBUG training (with rendering)...")
    cmd = build_train_command(config, render=True)
    return run_training(cmd)

def train_with_carla(config):
    """Auto-start CARLA and train"""
    print_info("Starting CARLA server and training...")
    print_warning("This will attempt to start CARLA automatically")
    cmd = build_train_command(config, render=False, start_carla=True)
    return run_training(cmd)

def train_quick_test(config):
    """Quick test (10k steps with rendering)"""
    print_info("Starting QUICK TEST (10k steps with rendering)...")
    test_config = config.copy()
    test_config['total_timesteps'] = 10000
    test_config['log_dir'] = 'tensorboard/test'
    test_config['save_freq'] = 5000
    cmd = build_train_command(test_config, render=True)
    return run_training(cmd)

# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(
        description='AutoRewardDrive Training Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  train    - Standard training (no rendering, fastest)
  debug    - Debug training (with rendering)
  auto     - Auto-start CARLA and train
  test     - Quick test (10k steps with rendering)

Examples:
  python run_train.py train
  python run_train.py debug
  python run_train.py test
        """
    )
    
    parser.add_argument('mode', nargs='?', default='train',
                       choices=['train', 'debug', 'auto', 'test'],
                       help='Training mode (default: train)')
    
    parser.add_argument('--host', default=CONFIG['host'],
                       help=f"CARLA host (default: {CONFIG['host']})")
    parser.add_argument('--port', type=int, default=CONFIG['port'],
                       help=f"CARLA port (default: {CONFIG['port']})")
    parser.add_argument('--device', default=CONFIG['device'],
                       help=f"Device (default: {CONFIG['device']})")
    
    args = parser.parse_args()
    
    # Update config with command line args
    config = CONFIG.copy()
    config['host'] = args.host
    config['port'] = args.port
    config['device'] = args.device
    
    print("=" * 60)
    print("  AutoRewardDrive Training Launcher")
    print("=" * 60)
    print()
    
    # Execute based on mode
    if args.mode in ['train', 'debug', 'test']:
        if not check_carla(config['host'], config['port']):
            print_error("Please start CARLA server first!")
            print_info("Run: ./CarlaUE4.sh -quality-level=Low -RenderOffScreen")
            return 1
    
    success = False
    if args.mode == 'train':
        success = train_standard(config)
    elif args.mode == 'debug':
        success = train_debug(config)
    elif args.mode == 'auto':
        success = train_with_carla(config)
    elif args.mode == 'test':
        success = train_quick_test(config)
    
    if success:
        print_success("Training completed!")
        return 0
    else:
        print_error("Training failed or was interrupted")
        return 1

if __name__ == '__main__':
    sys.exit(main())
