#!/usr/bin/env python3
"""
Training Progress Monitor
========================
Monitor the tiny DermoMamba training progress
"""

import os
import time
import json
from datetime import datetime

def find_latest_checkpoint_dir():
    """Find the latest tiny training checkpoint directory"""
    checkpoints_dir = "checkpoints"
    
    if not os.path.exists(checkpoints_dir):
        return None
    
    tiny_dirs = []
    for item in os.listdir(checkpoints_dir):
        if 'tiny_enhanced' in item:
            full_path = os.path.join(checkpoints_dir, item)
            if os.path.isdir(full_path):
                tiny_dirs.append((full_path, os.path.getmtime(full_path)))
    
    if tiny_dirs:
        return sorted(tiny_dirs, key=lambda x: x[1], reverse=True)[0][0]
    
    return None

def check_training_progress():
    """Check current training progress"""
    
    checkpoint_dir = find_latest_checkpoint_dir()
    
    if not checkpoint_dir:
        print("❌ No training checkpoint directory found")
        return None
    
    print(f"📂 Monitoring: {checkpoint_dir}")
    
    # Check for training log
    log_file = os.path.join(checkpoint_dir, 'training_log.json')
    model_file = os.path.join(checkpoint_dir, 'best_tiny_model.pth')
    
    progress_info = {
        'checkpoint_dir': checkpoint_dir,
        'log_exists': os.path.exists(log_file),
        'model_exists': os.path.exists(model_file),
        'dir_created': datetime.fromtimestamp(os.path.getmtime(checkpoint_dir)),
        'files_in_dir': os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else []
    }
    
    if progress_info['log_exists']:
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                progress_info['latest_log'] = log_data
        except Exception as e:
            progress_info['log_error'] = str(e)
    
    return progress_info

def monitor_training(check_interval=30):
    """Monitor training progress with periodic updates"""
    
    print("🔍 Tiny DermoMamba Training Monitor")
    print("=" * 50)
    print(f"Started monitoring at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Check interval: {check_interval} seconds")
    print()
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"🕐 [{current_time}] Check #{iteration}")
            print("-" * 30)
            
            progress = check_training_progress()
            
            if progress:
                print(f"📂 Directory: {os.path.basename(progress['checkpoint_dir'])}")
                print(f"📅 Created: {progress['dir_created'].strftime('%H:%M:%S')}")
                print(f"📄 Files: {len(progress['files_in_dir'])} items")
                
                if progress['files_in_dir']:
                    print(f"   Contents: {', '.join(progress['files_in_dir'])}")
                
                if progress['model_exists']:
                    model_path = os.path.join(progress['checkpoint_dir'], 'best_tiny_model.pth')
                    model_size = os.path.getsize(model_path) / (1024 * 1024)
                    model_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
                    print(f"✅ Model found: {model_size:.1f}MB, modified at {model_modified.strftime('%H:%M:%S')}")
                
                if 'latest_log' in progress:
                    log_data = progress['latest_log']
                    print(f"📊 Latest metrics:")
                    print(f"   Epoch: {log_data.get('epoch', 'Unknown')}")
                    
                    metrics = log_data.get('metrics', {})
                    if 'val_dice' in metrics and 'val_iou' in metrics:
                        print(f"   Validation Dice: {metrics['val_dice']:.4f}")
                        print(f"   Validation IoU: {metrics['val_iou']:.4f}")
                    
                    if 'learning_rate' in metrics:
                        print(f"   Learning Rate: {metrics['learning_rate']:.6f}")
                
                if not progress['log_exists'] and not progress['model_exists']:
                    print("⏳ Training in progress - no checkpoints saved yet")
                
            else:
                print("❌ No training directory found")
            
            print()
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n💥 Monitor error: {str(e)}")

if __name__ == "__main__":
    monitor_training()
