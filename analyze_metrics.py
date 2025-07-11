#!/usr/bin/env python3
"""
Metrics Parser for Mario DS AI Training
This script can be used to parse and analyze the metrics files generated during training.
"""

import csv
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_step_metrics(metrics_dir='metrics'):
    """Load step-by-step metrics from CSV"""
    step_file = os.path.join(metrics_dir, 'step_metrics.csv')
    if os.path.exists(step_file):
        return pd.read_csv(step_file)
    else:
        print(f"Step metrics file not found: {step_file}")
        return None

def load_episode_metrics(metrics_dir='metrics'):
    """Load episode metrics from CSV"""
    episode_file = os.path.join(metrics_dir, 'episode_metrics.csv')
    if os.path.exists(episode_file):
        return pd.read_csv(episode_file)
    else:
        print(f"Episode metrics file not found: {episode_file}")
        return None

def load_latest_snapshot(metrics_dir='metrics'):
    """Load the latest training snapshot"""
    snapshot_file = os.path.join(metrics_dir, 'latest_snapshot.json')
    if os.path.exists(snapshot_file):
        with open(snapshot_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Latest snapshot not found: {snapshot_file}")
        return None

def analyze_training_progress(metrics_dir='metrics'):
    """Analyze training progress and print summary"""
    print("=== Mario DS AI Training Analysis ===\n")
    
    # Load data
    step_df = load_step_metrics(metrics_dir)
    episode_df = load_episode_metrics(metrics_dir)
    snapshot = load_latest_snapshot(metrics_dir)
    
    # Analyze step metrics
    if step_df is not None and len(step_df) > 0:
        print("📊 Step Metrics Summary:")
        print(f"  Total steps recorded: {len(step_df)}")
        print(f"  Average reward per step: {step_df['reward'].mean():.4f}")
        print(f"  Recent average reward (last 100): {step_df['reward'].tail(100).mean():.4f}")
        
        if 'loss' in step_df.columns and step_df['loss'].notna().sum() > 0:
            loss_data = step_df['loss'].dropna()
            print(f"  Average loss: {loss_data.mean():.4f}")
            print(f"  Recent average loss (last 100): {loss_data.tail(100).mean():.4f}")
        print()
    
    # Analyze episode metrics
    if episode_df is not None and len(episode_df) > 0:
        print("🎮 Episode Metrics Summary:")
        print(f"  Total episodes: {len(episode_df)}")
        print(f"  Average episode reward: {episode_df['total_reward'].mean():.2f}")
        print(f"  Best episode reward: {episode_df['total_reward'].max():.2f}")
        print(f"  Recent average (last 10): {episode_df['total_reward'].tail(10).mean():.2f}")
        print(f"  Average max X position: {episode_df['max_x_pos'].mean():.4f}")
        print(f"  Best max X position: {episode_df['max_x_pos'].max():.4f}")
        print(f"  Levels completed: {episode_df['level_completed'].sum()}")
        print(f"  Level completion rate: {(episode_df['level_completed'].sum() / len(episode_df) * 100):.1f}%")
        
        # Death reason analysis
        death_reasons = episode_df['death_reason'].value_counts()
        print(f"\n  Death reasons:")
        for reason, count in death_reasons.items():
            percentage = (count / len(episode_df)) * 100
            print(f"    {reason}: {count} ({percentage:.1f}%)")
        print()
    
    # Snapshot analysis
    if snapshot is not None:
        print("📈 Latest Training State:")
        episode_info = snapshot.get('episode_info', {})
        print(f"  Current episode: {snapshot.get('episode', 'N/A')}")
        print(f"  Last episode reward: {episode_info.get('total_reward', 'N/A')}")
        print(f"  Last max X position: {episode_info.get('max_x_position', 'N/A')}")
        print(f"  Last death reason: {episode_info.get('death_reason', 'N/A')}")
        
        if 'recent_rewards' in snapshot and len(snapshot['recent_rewards']) > 0:
            recent_rewards = snapshot['recent_rewards']
            print(f"  Recent reward trend: {np.mean(recent_rewards):.4f}")
        
        if 'recent_losses' in snapshot and len(snapshot['recent_losses']) > 0:
            recent_losses = [l for l in snapshot['recent_losses'] if l is not None]
            if recent_losses:
                print(f"  Recent loss trend: {np.mean(recent_losses):.4f}")
        print()

def export_metrics_summary(metrics_dir='metrics', output_file='training_summary.json'):
    """Export a comprehensive training summary"""
    summary = {}
    
    # Load all data
    step_df = load_step_metrics(metrics_dir)
    episode_df = load_episode_metrics(metrics_dir)
    snapshot = load_latest_snapshot(metrics_dir)
    
    # Step metrics summary
    if step_df is not None and len(step_df) > 0:
        summary['step_metrics'] = {
            'total_steps': len(step_df),
            'avg_reward': float(step_df['reward'].mean()),
            'recent_avg_reward': float(step_df['reward'].tail(100).mean()),
            'reward_std': float(step_df['reward'].std())
        }
        
        if 'loss' in step_df.columns and step_df['loss'].notna().sum() > 0:
            loss_data = step_df['loss'].dropna()
            summary['step_metrics']['avg_loss'] = float(loss_data.mean())
            summary['step_metrics']['recent_avg_loss'] = float(loss_data.tail(100).mean())
    
    # Episode metrics summary
    if episode_df is not None and len(episode_df) > 0:
        summary['episode_metrics'] = {
            'total_episodes': len(episode_df),
            'avg_reward': float(episode_df['total_reward'].mean()),
            'best_reward': float(episode_df['total_reward'].max()),
            'recent_avg_reward': float(episode_df['total_reward'].tail(10).mean()),
            'avg_max_x_pos': float(episode_df['max_x_pos'].mean()),
            'best_max_x_pos': float(episode_df['max_x_pos'].max()),
            'levels_completed': int(episode_df['level_completed'].sum()),
            'completion_rate': float(episode_df['level_completed'].sum() / len(episode_df) * 100),
            'death_reasons': episode_df['death_reason'].value_counts().to_dict()
        }
    
    # Latest state
    if snapshot is not None:
        summary['latest_state'] = snapshot
    
    summary['generated_at'] = pd.Timestamp.now().isoformat()
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Training summary exported to: {output_file}")
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Mario DS AI training metrics")
    parser.add_argument('--metrics-dir', default='metrics', help='Directory containing metrics files')
    parser.add_argument('--export', action='store_true', help='Export summary to JSON file')
    parser.add_argument('--output', default='training_summary.json', help='Output file for summary')
    
    args = parser.parse_args()
    
    # Analyze training progress
    analyze_training_progress(args.metrics_dir)
    
    # Export summary if requested
    if args.export:
        export_metrics_summary(args.metrics_dir, args.output)
