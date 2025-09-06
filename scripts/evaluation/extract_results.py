"""
Extract best dice score from checkpoint
"""
import torch
import sys
sys.path.append('.')

def extract_checkpoint_metrics():
    print("="*70)
    print("Extracting Training Results from Checkpoints")
    print("="*70)
    
    # Check different checkpoint directories
    checkpoint_paths = [
        "checkpoints/optimized_complete_improved/best_model.ckpt",
        "checkpoints/optimized_complete_improved/best_model-v1.ckpt",
        "checkpoints/optimized_complete/best_model.ckpt",
        "checkpoints/optimized_complete/best_model-v1.ckpt"
    ]
    
    results = {}
    
    for checkpoint_path in checkpoint_paths:
        try:
            print(f"\nChecking: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract available information
            info = {}
            
            # Check for callback states
            if 'callbacks' in checkpoint:
                callbacks = checkpoint['callbacks']
                for callback_key, callback_data in callbacks.items():
                    if 'ModelCheckpoint' in callback_key:
                        if 'best_model_score' in callback_data:
                            info['best_dice'] = float(callback_data['best_model_score'])
                        if 'best_model_path' in callback_data:
                            info['best_model_path'] = callback_data['best_model_path']
                        if 'current_score' in callback_data:
                            info['current_score'] = float(callback_data['current_score'])
            
            # Check epoch and global step
            if 'epoch' in checkpoint:
                info['epoch'] = checkpoint['epoch']
            if 'global_step' in checkpoint:
                info['global_step'] = checkpoint['global_step']
            
            # Check learning rate
            if 'lr_schedulers' in checkpoint:
                info['lr_schedulers'] = len(checkpoint['lr_schedulers'])
            
            # Check optimizer state
            if 'optimizer_states' in checkpoint:
                info['optimizers'] = len(checkpoint['optimizer_states'])
            
            # Check model state dict size
            if 'state_dict' in checkpoint:
                info['model_parameters'] = len(checkpoint['state_dict'])
            
            # Store results
            results[checkpoint_path] = info
            
            # Print information
            if info:
                print(f"  ✅ Successfully loaded checkpoint")
                for key, value in info.items():
                    print(f"     {key}: {value}")
            else:
                print(f"  ❌ No metric information found")
                
        except Exception as e:
            print(f"  ❌ Error loading {checkpoint_path}: {e}")
    
    print("\n" + "="*70)
    print("TRAINING RESULTS SUMMARY")
    print("="*70)
    
    # Find best results
    best_dice = 0
    best_checkpoint = None
    
    for checkpoint_path, info in results.items():
        if 'best_dice' in info:
            dice_score = info['best_dice']
            if dice_score > best_dice:
                best_dice = dice_score
                best_checkpoint = checkpoint_path
            
            print(f"\n📊 {checkpoint_path}:")
            print(f"   🎯 Best Dice Score: {dice_score:.4f} ({dice_score*100:.2f}%)")
            if 'epoch' in info:
                print(f"   📈 Final Epoch: {info['epoch']}")
            if 'global_step' in info:
                print(f"   🔄 Total Steps: {info['global_step']}")
    
    if best_checkpoint:
        print(f"\n🏆 BEST OVERALL PERFORMANCE:")
        print(f"   🥇 Checkpoint: {best_checkpoint}")
        print(f"   🎯 Best Dice: {best_dice:.4f} ({best_dice*100:.2f}%)")
        
        # Compare with paper target
        paper_target = 0.91
        if best_dice >= paper_target:
            print(f"   ✅ ACHIEVED PAPER PERFORMANCE! ({paper_target*100:.0f}%+ target)")
        else:
            gap = paper_target - best_dice
            print(f"   📈 Gap to paper target: {gap:.4f} ({gap*100:.2f}% points)")
            print(f"   🎯 Paper target: {paper_target:.4f} ({paper_target*100:.0f}%)")
    
    else:
        print(f"\n❌ No dice scores found in checkpoints")
    
    return results

if __name__ == "__main__":
    results = extract_checkpoint_metrics()
