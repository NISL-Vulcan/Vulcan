#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple debugging script for dataset optimization.
"""

import os
import sys
import subprocess
import time

def test_optimization_script_direct():
    """Directly test optimization script."""
    print(" Direct optimization script test")
    print("=" * 50)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")
    
    # Check optimization script path
    script_path = os.path.join(current_dir, "auto_update_and_dynamic_ratio.py")
    print(f"Script path: {script_path}")
    
    if not os.path.exists(script_path):
        print(" Optimization script does not exist")
        return False
    
    print(" Optimization script exists")
    
    # Check dataset directory
    dataset_dir = os.path.join(current_dir, "dataset")
    if os.path.exists(dataset_dir):
        print(f" Dataset directory exists: {dataset_dir}")
        files = os.listdir(dataset_dir)
        print(f"   File list: {files}")
    else:
        print(f" Dataset directory does not exist: {dataset_dir}")
        return False
    
    # Check configs directory
    configs_dir = os.path.join(current_dir, "configs")
    if os.path.exists(configs_dir):
        print(f" Config directory exists: {configs_dir}")
        files = os.listdir(configs_dir)
        print(f"   File list: {files}")
    else:
        print(f" Config directory does not exist: {configs_dir}")
        return False
    
    # Try direct run for a few seconds
    print("\n Running optimization script (stop after 5 seconds)...")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # Start process
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=current_dir,
            env=env
        )
        
        print(f" Process started, PID: {process.pid}")
        
        # Read output for 5 seconds
        start_time = time.time()
        output_lines = []
        
        while time.time() - start_time < 5:
            line = process.stdout.readline()
            if line:
                line = line.rstrip()
                print(f" {line}")
                output_lines.append(line)
            else:
                time.sleep(0.1)
        
        # Terminate process
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        
        print("\n Output summary:")
        print(f"   • Total output lines: {len(output_lines)}")
        print(f"   • Process return code: {process.returncode}")
        
        if output_lines:
            print("   • First 5 output lines:")
            for i, line in enumerate(output_lines[:5]):
                print(f"     {i+1}: {line}")
        
        return len(output_lines) > 0
        
    except Exception as e:
        print(f" Error while running script: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_script_import():
    """Test importing optimization script."""
    print("\n Import test for optimization script")
    print("=" * 50)
    
    try:
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # Try import
        print("Importing auto_update_and_dynamic_ratio...")
        import auto_update_and_dynamic_ratio
        
        print(" Import succeeded")
        
        # Try creating instance
        print("Creating DynamicRatioTuner instance...")
        tuner = auto_update_and_dynamic_ratio.DynamicRatioTuner()
        print(" Instance created successfully")
        
        return True
        
    except Exception as e:
        print(f" Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(" Start debugging optimization script...")
    
    # Test 1: direct run
    direct_test = test_optimization_script_direct()
    
    # Test 2: import path
    import_test = test_optimization_script_import()
    
    print("\n Test results:")
    print(f"Direct run test: {'PASS' if direct_test else 'FAIL'}")
    print(f"Import test: {'PASS' if import_test else 'FAIL'}")
    
    if direct_test and import_test:
        print("\n Optimization script works as expected")
    else:
        print("\n Optimization script has issues and needs fixes")