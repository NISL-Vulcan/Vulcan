#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full diagnostics for dataset optimization issues.
"""

import os
import sys
import requests
import subprocess
import time
import json

def check_backend_service():
    """Check backend service."""
    print(" 1. Check backend service")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print(" Backend service is running")
            return True
        else:
            print(f" Backend service unhealthy, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f" Cannot connect to backend service: {e}")
        return False

def check_files():
    """Check required files."""
    print("\n 2. Check required files")
    print("-" * 30)
    
    files_to_check = [
        "auto_update_and_dynamic_ratio.py",
        "backend_server.py",
        "dataset/vulnerables.jsonl",
        "dataset/non-vulnerables.jsonl",
        "configs/regvd_reveal.yaml"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f" {file_path}")
        else:
            print(f" {file_path} - missing")
            all_exist = False
    
    return all_exist

def test_optimization_script():
    """Test optimization script."""
    print("\n 3. Test optimization script")
    print("-" * 30)
    
    try:
        # Try importing script
        import auto_update_and_dynamic_ratio
        print(" Script import succeeded")
        
        # Try creating instance
        tuner = auto_update_and_dynamic_ratio.DynamicRatioTuner()
        print(" Instance created successfully")
        
        return True
    except Exception as e:
        print(f" Script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test API endpoints."""
    print("\n 4. Test API endpoints")
    print("-" * 30)
    
    try:
        # Test start optimization API
        print("Testing optimization start API...")
        response = requests.post(
            "http://localhost:5000/api/start-dataset-optimization",
            json={"max_iterations": 1},  # test only 1 iteration
            timeout=10
        )
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if result.get("success"):
                job_id = result["job_id"]
                print(f" Optimization job started successfully, job ID: {job_id}")
                
                # Test status API
                print("\nTesting status API...")
                status_response = requests.get(f"http://localhost:5000/api/optimization-status/{job_id}", timeout=5)
                
                if status_response.status_code == 200:
                    status_result = status_response.json()
                    print(f"Status query succeeded: {json.dumps(status_result, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    print(f" Status query failed, status code: {status_response.status_code}")
                    return False
            else:
                print(f" Failed to start optimization: {result.get('error')}")
                return False
        else:
            print(f" Optimization start API failed, status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f" API test failed: {e}")
        return False

def main():
    """Main diagnostics entrypoint."""
    print(" Start diagnosing dataset optimization issues")
    print("=" * 50)
    
    # 1. Check backend service
    backend_ok = check_backend_service()
    
    # 2. Check files
    files_ok = check_files()
    
    # 3. Test optimization script
    script_ok = test_optimization_script()
    
    # 4. Test API endpoints
    api_ok = False
    if backend_ok and files_ok and script_ok:
        api_ok = test_api_endpoints()
    
    # Summary
    print("\n Diagnosis summary")
    print("=" * 50)
    print(f"Backend service: {'OK' if backend_ok else 'FAILED'}")
    print(f"Required files: {'COMPLETE' if files_ok else 'MISSING'}")
    print(f"Optimization script: {'OK' if script_ok else 'FAILED'}")
    print(f"API endpoints: {'OK' if api_ok else 'FAILED'}")
    
    if backend_ok and files_ok and script_ok and api_ok:
        print("\n All checks passed. Dataset optimization should work normally.")
        print("If frontend still hangs, the issue is likely on frontend side.")
    else:
        print("\n Issues detected, fixes are required.")
        if not backend_ok:
            print("- Start backend service: python start_services.py")
        if not files_ok:
            print("- Check whether required files exist")
        if not script_ok:
            print("- Check optimization script for runtime errors")
        if not api_ok:
            print("- Check API endpoint configuration")

if __name__ == "__main__":
    main() 