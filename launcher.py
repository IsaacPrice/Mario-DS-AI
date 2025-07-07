import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def launch_training():
    """Interactive launcher for training"""
    print("=== Mario DS AI Training Launcher ===\n")
    
    # Check if requirements are installed
    try:
        import torch
        import gym
        import matplotlib
    except ImportError:
        print("Some required packages are missing. Installing...")
        install_requirements()
    
    # Algorithm selection
    print("Choose RL Algorithm:")
    print("1. Rainbow DQN (Deep Q-Network with improvements)")
    print("2. PPO (Proximal Policy Optimization)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == "1":
            algorithm = "rainbow"
            break
        elif choice == "2":
            algorithm = "ppo"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Mode selection
    print(f"\nSelected algorithm: {algorithm.upper()}")
    print("Choose mode:")
    print("1. Train new model")
    print("2. Test existing model")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == "1":
            mode = "train"
            break
        elif choice == "2":
            mode = "test"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Additional parameters
    if mode == "train":
        episodes = input("\nNumber of episodes (default 1000): ").strip()
        episodes = episodes if episodes else "1000"
        
        save_interval = input("Save interval (default 100): ").strip()
        save_interval = save_interval if save_interval else "100"
        
        # Build command
        cmd = [
            sys.executable, "train_mario.py",
            "--algorithm", algorithm,
            "--mode", mode,
            "--episodes", episodes,
            "--save_interval", save_interval
        ]
    
    else:  # test mode
        model_path = input(f"\nModel path (default models/{algorithm}_best.pth): ").strip()
        if not model_path:
            model_path = f"models/{algorithm}_best.pth"
        
        test_episodes = input("Number of test episodes (default 5): ").strip()
        test_episodes = test_episodes if test_episodes else "5"
        
        # Build command
        cmd = [
            sys.executable, "train_mario.py",
            "--algorithm", algorithm,
            "--mode", mode,
            "--episodes", test_episodes,
            "--model_path", model_path
        ]
    
    # Launch training
    print(f"\nLaunching {mode} with {algorithm.upper()}...")
    print(f"Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop training/testing at any time.")
    print("-" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nTraining/Testing interrupted by user.")
    except FileNotFoundError:
        print("\nError: train_mario.py not found. Make sure you're in the correct directory.")
    except Exception as e:
        print(f"\nError occurred: {e}")

def quick_train_rainbow():
    """Quick start Rainbow DQN training"""
    cmd = [
        sys.executable, "train_mario.py",
        "--algorithm", "rainbow",
        "--mode", "train",
        "--episodes", "1000"
    ]
    subprocess.run(cmd)

def quick_train_ppo():
    """Quick start PPO training"""
    cmd = [
        sys.executable, "train_mario.py",
        "--algorithm", "ppo",
        "--mode", "train",
        "--episodes", "1000"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "rainbow":
            quick_train_rainbow()
        elif sys.argv[1] == "ppo":
            quick_train_ppo()
        elif sys.argv[1] == "install":
            install_requirements()
        else:
            print("Usage: python launcher.py [rainbow|ppo|install]")
    else:
        launch_training()
