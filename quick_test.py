#!/usr/bin/env python3
"""
Simplified Mario DS RL Training Script
"""

import os
import sys
import argparse
import time

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported")
    except ImportError as e:
        print(f"✗ NumPy failed: {e}")
        return False
        
    try:
        import gymnasium as gym
        print("✓ Gymnasium imported")
    except ImportError as e:
        print(f"✗ Gymnasium failed: {e}")
        return False
        
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported")
    except ImportError as e:
        print(f"✗ Matplotlib failed: {e}")
        return False
        
    print("Core imports successful!")
    return True

def test_torch():
    """Test PyTorch separately as it might be slow"""
    print("Testing PyTorch (may take a moment)...")
    try:
        import torch
        print(f"✓ PyTorch imported (version: {torch.__version__})")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"✗ PyTorch failed: {e}")
        return False

def test_environment():
    """Test environment creation without emulator"""
    print("Testing environment class...")
    try:
        # Test just the class definition without initialization
        from mario_env import MarioDSEnv
        print("✓ MarioDSEnv class imported successfully")
        return True
    except ImportError as e:
        print(f"✗ MarioDSEnv import failed: {e}")
        return False

def test_agents():
    """Test agent classes"""
    print("Testing agent classes...")
    try:
        from ppo_agent import PPOAgent
        print("✓ PPOAgent class imported successfully")
    except ImportError as e:
        print(f"✗ PPOAgent import failed: {e}")
        return False
        
    try:
        from rainbow_dqn import RainbowDQNAgent  
        print("✓ RainbowDQNAgent class imported successfully")
    except ImportError as e:
        print(f"✗ RainbowDQNAgent import failed: {e}")
        return False
        
    return True

def main():
    print("=" * 50)
    print("Mario DS AI Simplified Test")
    print("=" * 50)
    
    # Test core imports
    if not test_imports():
        print("Core imports failed. Please check your environment.")
        return
        
    # Test PyTorch
    if not test_torch():
        print("PyTorch failed. Training may not work properly.")
        return
        
    # Test environment
    if not test_environment():
        print("Environment test failed.")
        return
        
    # Test agents
    if not test_agents():
        print("Agent test failed.")
        return
        
    print("All tests passed! Ready for training.")
    print("\nTo start training:")
    print("python train_mario.py --algorithm ppo --episodes 100")
    print("or")
    print("python train_mario.py --algorithm rainbow_dqn --episodes 100")

if __name__ == "__main__":
    main()
