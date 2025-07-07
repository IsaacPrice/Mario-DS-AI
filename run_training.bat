@echo off
echo Mario DS AI Training Launcher
echo ============================
echo.
echo Choose your RL algorithm:
echo 1. Rainbow DQN
echo 2. PPO
echo 3. Test setup
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting Rainbow DQN training...
    python3.11 train_mario.py --algorithm rainbow --mode train --episodes 1000
) else if "%choice%"=="2" (
    echo Starting PPO training...
    python3.11 train_mario.py --algorithm ppo --mode train --episodes 1000
) else if "%choice%"=="3" (
    echo Running setup test...
    python3.11 test_setup.py
) else if "%choice%"=="4" (
    echo Goodbye!
    exit
) else (
    echo Invalid choice. Please run the script again.
)

pause
