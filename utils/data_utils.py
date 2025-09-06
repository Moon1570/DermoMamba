"""
Data Utilities for DermoMamba

Functions for data preprocessing, splitting, and dataset management.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil

def create_data_splits(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    output_dir: str = "splits",
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create train/validation splits for dataset
    
    Args:
        data_dir: Directory containing images
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        output_dir: Directory to save split files
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: Dictionary with train and val file lists
    """
    random.seed(random_seed)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for file in os.listdir(data_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    
    # Create splits
    train_files = image_files[:train_end]
    val_files = image_files[train_end:]
    
    splits = {
        'train': train_files,
        'val': val_files
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits to files
    for split_name, file_list in splits.items():
        split_file = os.path.join(output_dir, f"isic2018_{split_name}.txt")
        with open(split_file, 'w') as f:
            for filename in file_list:
                f.write(f"{filename}\n")
        print(f"✅ Saved {len(file_list)} files to {split_file}")
    
    return splits

def load_data_splits(splits_dir: str = "splits") -> Dict[str, List[str]]:
    """
    Load existing data splits from files
    
    Args:
        splits_dir: Directory containing split files
    
    Returns:
        dict: Dictionary with loaded splits
    """
    splits = {}
    split_files = ['train', 'val', 'test']
    
    for split in split_files:
        split_file = os.path.join(splits_dir, f"isic2018_{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits[split] = [line.strip() for line in f if line.strip()]
            print(f"✅ Loaded {len(splits[split])} {split} files")
        else:
            print(f"⚠️ Split file not found: {split_file}")
    
    return splits

def validate_dataset_structure(data_dir: str) -> Dict[str, any]:
    """
    Validate dataset directory structure
    
    Args:
        data_dir: Root data directory
    
    Returns:
        dict: Validation results
    """
    validation = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    required_dirs = ['train_images', 'train_masks', 'val_images']
    optional_dirs = ['test_images']
    
    # Check required directories
    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.exists(dir_path):
            validation['valid'] = False
            validation['errors'].append(f"Missing required directory: {dir_name}")
        else:
            file_count = len(os.listdir(dir_path))
            validation['statistics'][dir_name] = file_count
    
    # Check optional directories
    for dir_name in optional_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.exists(dir_path):
            file_count = len(os.listdir(dir_path))
            validation['statistics'][dir_name] = file_count
        else:
            validation['warnings'].append(f"Optional directory not found: {dir_name}")
    
    # Check image-mask pairs
    if 'train_images' in validation['statistics'] and 'train_masks' in validation['statistics']:
        img_count = validation['statistics']['train_images']
        mask_count = validation['statistics']['train_masks']
        if img_count != mask_count:
            validation['valid'] = False
            validation['errors'].append(f"Image-mask count mismatch: {img_count} images, {mask_count} masks")
    
    return validation

def fix_data_splits(splits_dir: str = "splits", backup: bool = True) -> bool:
    """
    Fix common issues with data split files
    
    Args:
        splits_dir: Directory containing split files
        backup: Whether to create backup before fixing
    
    Returns:
        bool: True if fixes were applied successfully
    """
    try:
        if backup:
            backup_dir = f"{splits_dir}_backup"
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(splits_dir, backup_dir)
            print(f"✅ Created backup at {backup_dir}")
        
        split_files = ['train', 'val', 'test']
        fixes_applied = 0
        
        for split in split_files:
            split_file = os.path.join(splits_dir, f"isic2018_{split}.txt")
            if os.path.exists(split_file):
                # Read and clean lines
                with open(split_file, 'r') as f:
                    lines = f.readlines()
                
                # Remove empty lines and strip whitespace
                clean_lines = []
                for line in lines:
                    cleaned = line.strip()
                    if cleaned and not cleaned.startswith('#'):
                        clean_lines.append(cleaned)
                
                # Write back if changes were made
                if len(clean_lines) != len(lines) or any(line.strip() != clean_lines[i] for i, line in enumerate(lines[:len(clean_lines)])):
                    with open(split_file, 'w') as f:
                        for line in clean_lines:
                            f.write(f"{line}\n")
                    fixes_applied += 1
                    print(f"✅ Fixed {split_file}: {len(clean_lines)} clean lines")
        
        if fixes_applied > 0:
            print(f"✅ Applied fixes to {fixes_applied} split files")
        else:
            print("✅ No fixes needed, split files are clean")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing splits: {e}")
        return False

def get_dataset_statistics(data_dir: str) -> Dict[str, any]:
    """
    Get comprehensive dataset statistics
    
    Args:
        data_dir: Dataset directory
    
    Returns:
        dict: Dataset statistics
    """
    stats = {
        'total_images': 0,
        'total_masks': 0,
        'directories': {},
        'file_extensions': {},
        'size_info': {}
    }
    
    for root, dirs, files in os.walk(data_dir):
        dir_name = os.path.basename(root)
        if dir_name and dir_name != os.path.basename(data_dir):
            stats['directories'][dir_name] = len(files)
            
            # Count by extension
            for file in files:
                ext = Path(file).suffix.lower()
                stats['file_extensions'][ext] = stats['file_extensions'].get(ext, 0) + 1
                
                if 'image' in dir_name.lower():
                    stats['total_images'] += 1
                elif 'mask' in dir_name.lower():
                    stats['total_masks'] += 1
    
    return stats

if __name__ == "__main__":
    # Example usage
    print("🔧 Data Utilities Demo")
    
    # Check if data directory exists
    data_dir = "data/ISIC2018"
    if os.path.exists(data_dir):
        validation = validate_dataset_structure(data_dir)
        print(f"Dataset valid: {validation['valid']}")
        if validation['errors']:
            print("Errors:", validation['errors'])
        print("Statistics:", validation['statistics'])
    else:
        print(f"Data directory not found: {data_dir}")
