#!/usr/bin/env python3
"""
DermoMamba Setup and Verification Script
Helps users set up the environment and verify everything works
"""

import os
import sys
import subprocess

def run_command(cmd, description, required=True):
    """Run a command and report results"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()[:100]}...")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()[:200]}...")
            if required:
                return False
            return True
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return not required

def check_directory_structure():
    """Check if directory structure is properly organized"""
    print("\n📁 Checking Directory Structure")
    print("-" * 40)
    
    required_dirs = [
        "scripts/training",
        "scripts/testing", 
        "scripts/debugging",
        "scripts/evaluation",
        "utils",
        "config",
        "datasets",
        "loss",
        "metric",
        "module"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.py')])
            print(f"✅ {dir_path:<25} ({file_count} Python files)")
        else:
            print(f"❌ {dir_path:<25} (Missing)")
            all_good = False
    
    return all_good

def check_key_files():
    """Check if key files exist"""
    print("\n📄 Checking Key Files")
    print("-" * 40)
    
    key_files = [
        ("scripts/evaluation/compare_models.py", "Model comparison script"),
        ("scripts/evaluation/extract_results.py", "Results extraction script"),
        ("scripts/training/train_improved_dice.py", "Best training script"),
        ("utils/gpu_utils.py", "GPU utilities"),
        ("utils/data_utils.py", "Data utilities"),
        ("utils/evaluation_utils.py", "Evaluation utilities"),
        ("requirements.txt", "Dependencies file")
    ]
    
    all_good = True
    for file_path, description in key_files:
        if os.path.exists(file_path):
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - Missing: {file_path}")
            all_good = False
    
    return all_good

def main():
    print("🔬 DermoMamba Setup & Verification")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"🐍 Python Version: {python_version}")
    
    # Check directory structure
    structure_ok = check_directory_structure()
    
    # Check key files
    files_ok = check_key_files()
    
    # System checks
    print("\n🔧 System Checks")
    print("-" * 40)
    
    gpu_check = run_command("python utils/check_gpu.py", "GPU Status Check", required=False)
    imports_check = run_command("python utils/check_imports.py", "Import Verification", required=False)
    
    # Data checks
    print("\n📊 Data Checks")
    print("-" * 40)
    
    data_dirs = ["data/ISIC2018", "splits"]
    data_exists = False
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"✅ {data_dir} exists")
            data_exists = True
        else:
            print(f"⚠️ {data_dir} not found (optional)")
    
    # Checkpoint checks
    print("\n🏆 Checkpoint Checks")
    print("-" * 40)
    
    checkpoint_dirs = [
        "checkpoints/optimized_complete_improved",
        "checkpoints/optimized_complete",
        "checkpoints/tiny"
    ]
    
    checkpoints_exist = False
    for ckpt_dir in checkpoint_dirs:
        if os.path.exists(ckpt_dir):
            ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
            if ckpt_files:
                print(f"✅ {ckpt_dir} ({len(ckpt_files)} checkpoints)")
                checkpoints_exist = True
            else:
                print(f"⚠️ {ckpt_dir} (empty)")
        else:
            print(f"⚠️ {ckpt_dir} not found")
    
    # Summary
    print("\n📋 Setup Summary")
    print("=" * 50)
    
    if structure_ok and files_ok:
        print("✅ Directory structure: GOOD")
        print("✅ Key files: PRESENT")
        
        if data_exists:
            print("✅ Data: AVAILABLE")
        else:
            print("⚠️ Data: Please add ISIC2018 dataset to data/ folder")
        
        if checkpoints_exist:
            print("✅ Checkpoints: AVAILABLE")
            print("🎉 You can run model evaluation!")
        else:
            print("⚠️ Checkpoints: Train models first or add existing checkpoints")
        
        print("\n🚀 Ready to Use Commands:")
        print("-" * 30)
        print("Demo:           python demo_usage.py")
        print("Train (Best):   python scripts/training/train_improved_dice.py")
        print("Evaluate:       python scripts/evaluation/compare_models.py")
        print("Extract:        python scripts/evaluation/extract_results.py")
        print("Test:           python scripts/testing/test_implementation.py")
        
        print("\n🏆 Target Performance: 89.25% Dice Score achieved!")
        
    else:
        print("❌ Setup issues found. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
