#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script.
"""

import os
import sys
import subprocess
import time

def quick_test():
    print(" Quick diagnosis for dataset optimization issues")
    print("=" * 40)
    
    # 1. Check file
    script_path = "auto_update_and_dynamic_ratio.py"
    if not os.path.exists(script_path):
        print(" Optimization script does not exist")
        return
    
    print(" Optimization script exists")
    
    # 2. Try running script (up to 10 seconds)
    print("\n Trying to run script...")
    
    try:
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        
        print(f"Process PID: {process.pid}")
        
        # Read output for 10 seconds
        start_time = time.time()
        output = []
        
        while time.time() - start_time < 10:
            line = process.stdout.readline()
            if line:
                line = line.rstrip()
                print(f" {line}")
                output.append(line)
            else:
                time.sleep(0.1)
        
        # Terminate process
        process.terminate()
        process.wait(timeout=3)
        
        print("\n Result:")
        print(f"Output lines: {len(output)}")
        print(f"Return code: {process.returncode}")
        
        if len(output) == 0:
            print(" No output detected; script may be stuck")
        else:
            print(" Script produced output and appears to work")
            
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    quick_test() 