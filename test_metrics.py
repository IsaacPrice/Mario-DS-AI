#!/usr/bin/env python3
"""
Test script to verify metrics logging functionality
"""

import numpy as np
import os
import sys
import time
from frameDisplay import FrameDisplay

def test_metrics_logging():
    """Test the metrics logging functionality"""
    print("🧪 Testing Mario DS AI Metrics Logging")
    print("=" * 50)
    
    # Initialize frame display
    frame_display = FrameDisplay()
    
    # Simulate some training data
    print("\n📝 Simulating training data...")
    
    # Simulate step data
    for step in range(20):
        reward = np.random.normal(0.1, 0.05)  # Random step reward
        loss = np.random.normal(1.0, 0.2) if step > 5 else None  # Random loss after warmup
        
        frame_display._log_step_data(reward, loss)
        print(f"  Step {step+1}: reward={reward:.4f}, loss={loss:.4f if loss else 'None'}")
        time.sleep(0.1)  # Small delay to simulate real training
    
    # Simulate episode data
    print("\n🎮 Simulating episode completion...")
    episode_info = {
        'episode': 1,
        'total_reward': 5.2,
        'max_x_position': 0.45,
        'final_x_position': 0.43,
        'episode_duration': 45.3,
        'total_frames': 2700,
        'total_actions': 540,
        'death_reason': 'enemy_or_pit',
        'level_completed': False,
        'action_distribution': {0: 50, 1: 200, 2: 150, 3: 80, 4: 60},
        'stuck_count': 15,
        'last_progress_step': 480,
        'average_progress_per_step': 0.00083
    }
    
    frame_display.update_episode_data(episode_info)
    print(f"  Episode 1 logged: reward={episode_info['total_reward']}, max_x={episode_info['max_x_position']}")
    
    # Simulate another episode
    episode_info_2 = {
        'episode': 2,
        'total_reward': 8.7,
        'max_x_position': 0.62,
        'final_x_position': 0.58,
        'episode_duration': 62.1,
        'total_frames': 3720,
        'total_actions': 744,
        'death_reason': 'timeout',
        'level_completed': False,
        'action_distribution': {0: 30, 1: 300, 2: 200, 3: 120, 4: 94},
        'stuck_count': 8,
        'last_progress_step': 720,
        'average_progress_per_step': 0.00083
    }
    
    frame_display.update_episode_data(episode_info_2)
    print(f"  Episode 2 logged: reward={episode_info_2['total_reward']}, max_x={episode_info_2['max_x_position']}")
    
    # Test metrics summary
    print("\n📊 Current metrics summary:")
    summary = frame_display.get_metrics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Close frame display
    frame_display.close()
    
    # Check if files were created
    print("\n📁 Checking generated files:")
    metrics_files = [
        'metrics/step_metrics.csv',
        'metrics/episode_metrics.csv',
        'metrics/episode_1_snapshot.json',
        'metrics/episode_2_snapshot.json',
        'metrics/latest_snapshot.json'
    ]
    
    for file_path in metrics_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} (not found)")
    
    print("\n🎉 Metrics logging test completed!")
    print("\nYou can now run 'python analyze_metrics.py' to view the results.")

if __name__ == "__main__":
    test_metrics_logging()
