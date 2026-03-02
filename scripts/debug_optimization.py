#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug dataset optimization workflow.
"""

import os
import sys
import subprocess

def check_environment():
    """Check local environment."""
    print(" Checking environment...")
    
    # Check Python version
    print(f" Python version: {sys.version}")
    print(f" Python executable: {sys.executable}")
    
    # Check current directory
    current_dir = os.getcwd()
    print(f" Current directory: {current_dir}")
    
    # Check required files
    required_files = [
        "backend_server.py",
        "auto_update_and_dynamic_ratio.py"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f" {file} exists")
        else:
            print(f" {file} does not exist")
    
    # Check dataset directory
    dataset_dir = "dataset"
    if os.path.exists(dataset_dir):
        print(f" Dataset directory exists: {dataset_dir}")
        dataset_files = os.listdir(dataset_dir)
        print(f" Dataset files: {dataset_files}")
    else:
        print(f" Dataset directory does not exist: {dataset_dir}")
    
    # Check configs directory
    configs_dir = "configs"
    if os.path.exists(configs_dir):
        print(f" Config directory exists: {configs_dir}")
        config_files = [f for f in os.listdir(configs_dir) if f.endswith('.yaml')]
        print(f" Config files: {config_files}")
    else:
        print(f" Config directory does not exist: {configs_dir}")

def test_optimization_script():
    """Test optimization script."""
    print("\n Testing optimization script...")
    
    try:
        # Run optimization script directly
        cmd = [sys.executable, "auto_update_and_dynamic_ratio.py"]
        print(f" Running command: {' '.join(cmd)}")
        
        # Run script and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ, PYTHONUNBUFFERED='1')
        )
        
        print(" Script output:")
        print("-" * 50)
        
        # Read first few lines
        line_count = 0
        for line in process.stdout:
            print(line.rstrip())
            line_count += 1
            if line_count >= 20:  # show first 20 lines only
                print("... (output truncated)")
                break
        
        # Wait for process completion
        return_code = process.wait()
        print("-" * 50)
        print(f" Script return code: {return_code}")
        
        if return_code == 0:
            print(" Optimization script succeeded")
        else:
            print(" Optimization script failed")
            
    except Exception as e:
        print(f" Exception while running optimization script: {e}")

def main():
    """Main entrypoint."""
    print(" Debugging dataset optimization workflow")
    print("=" * 60)
    
    # Check environment
    check_environment()
    
    # Test optimization script
    test_optimization_script()
    
    print("\n Debugging complete")
    print("\n Suggestions:")
    print("1. If the optimization script succeeds, environment setup is likely correct")
    print("2. If it fails, inspect the error output")
    print("3. Ensure dataset files exist and formats are valid")
    print("4. Ensure config files exist and formats are valid")

if __name__ == "__main__":
    main() 