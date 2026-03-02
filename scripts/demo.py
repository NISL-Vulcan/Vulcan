#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vulcan-Detection frontend/backend integration demo script.
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

def print_header():
    """Print demo header."""
    print("=" * 60)
    print("vulcan-Detection Frontend/Backend Integration Demo")
    print("=" * 60)
    print()

def check_backend_status():
    """Check backend server status."""
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print(" Backend server is running")
            return True
        else:
            print(" Backend server returned an unexpected response")
            return False
    except Exception as e:
        print(f" Cannot connect to backend server: {e}")
        print("Start backend first: python start_backend.py")
        return False

def demo_get_models():
    """Demo: fetch available models."""
    print("\n Fetching available model list...")
    try:
        response = requests.get("http://localhost:5000/api/models")
        data = response.json()
        
        if data.get("success"):
            models = data.get("models", [])
            print(f" Found {len(models)} available models:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
        else:
            print(f" Failed to fetch models: {data.get('error')}")
    except Exception as e:
        print(f" Request failed: {e}")

def demo_get_datasets():
    """Demo: fetch available datasets."""
    print("\n Fetching available dataset list...")
    try:
        response = requests.get("http://localhost:5000/api/datasets")
        data = response.json()
        
        if data.get("success"):
            datasets = data.get("datasets", [])
            print(f" Found {len(datasets)} available datasets:")
            for i, dataset in enumerate(datasets, 1):
                print(f"  {i}. {dataset}")
        else:
            print(f" Failed to fetch datasets: {data.get('error')}")
    except Exception as e:
        print(f" Request failed: {e}")

def demo_generate_config():
    """Demo: generate a config file."""
    print("\n Config generation demo...")
    
    config_request = {
        "model_name": "DeepWuKong",
        "dataset_name": "DWK_Dataset",
        "device": "cuda",
        "batch_size": 64,
        "epochs": 20,
        "learning_rate": 0.001,
        "eval_interval": 5,
        "save_dir": "demo_output"
    }
    
    print(" Config parameters:")
    for key, value in config_request.items():
        print(f"  {key}: {value}")
    
    try:
        response = requests.post(
            "http://localhost:5000/api/generate-config",
            json=config_request
        )
        data = response.json()
        
        if data.get("success"):
            print(f" {data.get('message')}")
            print(f" Config file path: {data.get('config_path')}")
            print(f" Config ID: {data.get('config_id')}")
            
            # Show selected config fields
            config = data.get('config', {})
            print("\n Config preview:")
            print(f"  Device: {config.get('DEVICE')}")
            print(f"  Model: {config.get('MODEL', {}).get('NAME')}")
            print(f"  Dataset: {config.get('DATASET', {}).get('NAME')}")
            print(f"  Epochs: {config.get('TRAIN', {}).get('EPOCHS')}")
            print(f"  Batch size: {config.get('TRAIN', {}).get('BATCH_SIZE')}")
            
            return data.get('config_id')
        else:
            print(f" Config generation failed: {data.get('error')}")
            return None
    except Exception as e:
        print(f" Request failed: {e}")
        return None

def demo_start_training(config_id):
    """Demo: start training."""
    if not config_id:
        print("\n Skip training demo - invalid config ID")
        return None
    
    print("\n Training startup demo...")
    
    try:
        response = requests.post(
            "http://localhost:5000/api/start-training",
            json={"config_id": config_id}
        )
        data = response.json()
        
        if data.get("success"):
            job_id = data.get('job_id')
            print(f" {data.get('message')}")
            print(f" Job ID: {job_id}")
            return job_id
        else:
            print(f" Failed to start training: {data.get('error')}")
            return None
    except Exception as e:
        print(f" Request failed: {e}")
        return None

def demo_monitor_training(job_id):
    """Demo: monitor training."""
    if not job_id:
        print("\n Skip training monitor demo - invalid job ID")
        return
    
    print("\n Training monitor demo...")
    print("Monitoring training progress for 30 seconds...")
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < 30:  # monitor for 30 seconds
        try:
            response = requests.get(f"http://localhost:5000/api/training-status/{job_id}")
            data = response.json()
            
            if data.get("success"):
                status = data.get('status')
                progress = data.get('progress', 0)
                current_epoch = data.get('current_epoch', 0)
                total_epochs = data.get('total_epochs', 0)
                metrics = data.get('metrics', {})
                
                # Print only when status changes
                if status != last_status:
                    print(f"\n Training status: {status}")
                    if status == 'running':
                        print(f" Progress: {progress:.1f}%")
                        print(f" Epoch: {current_epoch}/{total_epochs}")
                        if metrics.get('loss'):
                            print(f" Loss: {metrics['loss']:.4f}")
                        if metrics.get('accuracy'):
                            print(f" Accuracy: {metrics['accuracy']:.4f}")
                    last_status = status
                
                # Stop monitor when training completes or fails
                if status in ['completed', 'failed']:
                    print(f"\n Training {status}: monitoring finished")
                    if status == 'completed':
                        print(" Training completed successfully")
                    else:
                        print(" Training failed")
                    break
                    
            else:
                print(f" Failed to fetch status: {data.get('error')}")
                break
                
        except Exception as e:
            print(f" Monitor request failed: {e}")
            break
        
        time.sleep(3)  # query every 3 seconds
    
    print("\n Monitor demo finished")

def demo_command_examples():
    """Demo: natural language command examples."""
    print("\n Natural language command examples:")
    print("You can use commands like the following in the frontend UI:")
    print()
    
    examples = [
        "Generate config file, model: DeepWuKong, dataset: DWK_Dataset, device: cuda, epochs: 20, batch_size: 64",
        "Generate config file, model: Devign, dataset: Devign_Partial, learning_rate: 0.0001",
        "Generate config file, model: IVDetect, device: cpu, quick mode",
        "Start training",
        "Show training status",
        "Show training progress"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"  {i}. \"{example}\"")

def main():
    """Main entrypoint."""
    print_header()
    
    # Check backend status
    if not check_backend_status():
        return
    
    print("\n Starting feature demo...")
    
    # Demo: list models and datasets
    demo_get_models()
    demo_get_datasets()
    
    # Demo: generate config file
    config_id = demo_generate_config()
    
    # Ask whether to run training demo
    print("\n" + "="*50)
    print("  Training demo notes:")
    print("This will start a real training process and may take time.")
    print("The monitor runs for 30 seconds and then exits automatically.")
    print("="*50)
    
    response = input("\nContinue with training demo? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        # Demo: start training
        job_id = demo_start_training(config_id)
        
        # Demo: monitor training
        demo_monitor_training(job_id)
    else:
        print("  Skip training demo")
    
    # Show command examples
    demo_command_examples()
    
    print("\n" + "="*60)
    print(" Demo completed")
    print()
    print("Next steps:")
    print("1. Open frontend UI for visual operations")
    print("2. Call APIs programmatically")
    print("3. Inspect generated config files in: generated_configs/")
    print("="*60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Demo interrupted by user")
    except Exception as e:
        print(f"\n Error during demo: {e}")
        sys.exit(1) 