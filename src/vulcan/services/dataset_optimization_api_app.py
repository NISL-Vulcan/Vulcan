#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENAPIEN
ENbackend_server.pyEN
"""

import threading
import subprocess
import sys
import os
import uuid
from datetime import datetime
import json
import yaml
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
import time
import re
from typing import Dict, Any, List, Optional

# EN
optimization_jobs = {}

class OptimizationJob:
    """EN"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0
        self.current_iteration = 0
        self.total_iterations = 15
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.process = None
        self.best_ratio = 0.0
        self.best_f1_score = 0.0
        self.metrics = {}
        # EN
        self.log_file_path = f"logs/optimization_{job_id}.log"
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """EN"""
        log_dir = os.path.dirname(self.log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def add_log(self, message: str):
        """EN"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # EN
        self.logs.append(log_entry)
        
        # EN
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            logging.error(f"EN: {e}")
        
        # EN
        if len(self.logs) > 2000:
            self.logs = self.logs[-1500:]
    
    def get_logs(self, limit: int = None) -> List[str]:
        """EN,EN"""
        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    file_logs = f.readlines()
                    file_logs = [log.strip() for log in file_logs if log.strip()]
                    
                    if limit:
                        return file_logs[-limit:]
                    return file_logs
        except Exception as e:
            logging.error(f"EN: {e}")
        
        if limit:
            return self.logs[-limit:]
        return self.logs.copy()
    
    def get_recent_logs(self, count: int = 50) -> List[str]:
        """EN"""
        return self.get_logs(limit=count)
    
    def get_full_logs(self) -> List[str]:
        """EN"""
        return self.get_logs()
    
    def get_duration(self) -> str:
        """EN"""
        if not self.start_time:
            return "EN"
        
        if not self.end_time:
            end_time = datetime.now()
        else:
            end_time = datetime.fromisoformat(self.end_time)
        
        start_time = datetime.fromisoformat(self.start_time)
        duration = end_time - start_time
        
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        
        if hours > 0:
            return f"{hours}EN{minutes}EN{seconds}EN"
        elif minutes > 0:
            return f"{minutes}EN{seconds}EN"
        else:
            return f"{seconds}EN"
    
    def get_log_file_path(self) -> str:
        """EN"""
        return self.log_file_path

def run_dataset_optimization(job: OptimizationJob):
    """EN"""
    try:
        job.status = "running"
        job.start_time = datetime.now().isoformat()
        
        # EN
        startup_msg = f"\n{'='*80}\n vulcan EN\n{'='*80}"
        print(startup_msg)
        job.add_log(startup_msg)
        
        startup_info = f" EN:\n   • ENID: {job.job_id}\n   • EN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(startup_info)
        job.add_log(startup_info)
        
        # EN
        current_dir = os.path.dirname(__file__)
        project_root = current_dir
        
        print(f"\n EN:")
        print(f"   • EN: {current_dir}")
        print(f"   • EN: {project_root}")
        print(f"   • PythonEN: {sys.version.split()[0]}")
        print(f"   • PythonEN: {sys.executable}")
        job.add_log(f" EN: {project_root}")
        
        # EN
        optimization_script = os.path.join(project_root, "auto_update_and_dynamic_ratio.py")
        
        # EN
        print(f"\n EN:")
        
        if not os.path.exists(optimization_script):
            error_msg = f" EN: {optimization_script}"
            print(error_msg)
            job.add_log(error_msg)
            job.status = "failed"
            return
        else:
            print(f"    EN: {optimization_script}")
            job.add_log(f" EN: {optimization_script}")
        
        # EN
        dataset_dir = os.path.join(project_root, "dataset")
        print(f"\n EN:")
        
        if os.path.exists(dataset_dir):
            dataset_files = ["vulnerables.jsonl", "non-vulnerables.jsonl"]
            for dataset_file in dataset_files:
                file_path = os.path.join(dataset_dir, dataset_file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"    {dataset_file}: {file_size:,} bytes")
                    job.add_log(f" EN: {dataset_file} ({file_size:,} bytes)")
                else:
                    print(f"   ️  {dataset_file}: EN")
                    job.add_log(f"️ EN: {dataset_file}")
        else:
            print(f"   ️ EN: {dataset_dir}")
            job.add_log(f"️ EN: {dataset_dir}")
        
        # ENGPU
        print(f"\n EN:")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"    GPU: {gpu_name} ({gpu_count}EN)")
                print(f"    GPUEN: {gpu_memory:.1f} GB")
                job.add_log(f" GPU: {gpu_name} ({gpu_count}EN, {gpu_memory:.1f}GB)")
            else:
                print(f"   ️  GPU: EN,ENCPU")
                job.add_log(f"️ GPUEN,ENCPU")
        except ImportError:
            print(f"   ️  PyTorchEN,ENGPU")
            job.add_log(f"️ PyTorchEN,ENGPU")
        
        # EN
        cmd = [sys.executable, optimization_script]
        
        print(f"\n EN...")
        print(f"   EN: {' '.join(cmd)}")
        job.add_log(f" EN: {' '.join(cmd)}")
        
        # EN
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=project_root
        )
        
        job.process = process
        
        # EN
        patterns = {
            'iteration': r'EN(\d+)EN',  # EN "EN1EN"
            'ratio': r'EN\s+([0-9.]+)',   # EN "EN 0.5"
            'f1_score': r'F1EN:\s+([0-9.]+)',  # EN "F1EN: 0.8234"
            'best_ratio': r'EN:\s+([0-9.]+)',  # EN "EN: 0.6"
            'best_f1': r'F1EN:\s+([0-9.]+)',  # EN "F1EN: 0.8234"
            'progress': r'EN:\s*(\d+)%',  # EN "EN: 50%"
            'completed': r'EN|EN',  # EN
            'failed': r'EN|EN|Error|Exception|EN'  # EN
        }
        
        current_ratio = 0.0  # EN
        current_f1 = 0.0     # ENF1EN
        
        for line in process.stdout:
            print(line, end='')  # EN
            job.add_log(line.strip())  # EN
            
            # EN
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    if key == 'iteration':
                        job.current_iteration = int(match.group(1))
                        job.progress = min(100, int((job.current_iteration / job.total_iterations) * 100))
                        print(f" EN: {job.progress}% (EN{job.current_iteration}EN)")
                        job.add_log(f" EN: {job.progress}% (EN{job.current_iteration}EN)")
                    elif key == 'ratio':
                        current_ratio = float(match.group(1))
                        print(f" EN: {current_ratio}")
                        job.add_log(f" EN: {current_ratio}")
                    elif key == 'f1_score':
                        current_f1 = float(match.group(1))
                        print(f" ENF1EN: {current_f1}")
                        job.add_log(f" ENF1EN: {current_f1}")
                        if current_f1 > job.best_f1_score:
                            job.best_f1_score = current_f1
                            job.best_ratio = current_ratio
                            print(f" EN: EN={job.best_ratio:.3f}, F1={job.best_f1_score:.4f}")
                            job.add_log(f" EN: EN={job.best_ratio:.3f}, F1={job.best_f1_score:.4f}")
                    elif key == 'best_ratio':
                        job.best_ratio = float(match.group(1))
                        print(f" EN: {job.best_ratio}")
                        job.add_log(f" EN: {job.best_ratio}")
                    elif key == 'progress':
                        # EN
                        progress_percent = int(match.group(1))
                        job.progress = progress_percent
                        print(f" EN: {progress_percent}%")
                        job.add_log(f" EN: {progress_percent}%")
                    elif key == 'completed':
                        job.status = "completed"
                        job.progress = 100
                        print(" EN")
                        job.add_log(" EN")
                    elif key == 'failed':
                        job.status = "failed"
                        print(" EN")
                        job.add_log(" EN")
        
        # EN
        return_code = process.wait()
        job.end_time = datetime.now().isoformat()
        
        if return_code == 0 and job.status != "failed":
            job.status = "completed"
            job.progress = 100
            
            completion_msg = f"\n EN!"
            completion_msg += f"\n EN:"
            completion_msg += f"\n   • EN: {job.best_ratio:.3f}"
            completion_msg += f"\n   • ENF1EN: {job.best_f1_score:.4f}"
            completion_msg += f"\n   • EN: {job.current_iteration}"
            completion_msg += f"\n   • EN: {job.get_duration()}"
            
            print(completion_msg)
            job.add_log(completion_msg)
            
            # EN
            job.metrics = {
                "best_ratio": job.best_ratio,
                "best_f1_score": job.best_f1_score,
                "total_iterations": job.current_iteration,
                "duration": job.get_duration()
            }
        else:
            job.status = "failed"
            error_msg = f"\n EN!EN: {return_code}"
            print(error_msg)
            job.add_log(error_msg)
            
    except Exception as e:
        job.status = "failed"
        job.end_time = datetime.now().isoformat()
        error_msg = f"EN: {str(e)}"
        print(error_msg)
        job.add_log(error_msg)
        logging.error(f"EN: {e}")
        import traceback
        job.add_log(traceback.format_exc())

def start_dataset_optimization_api(max_iterations: int = 15) -> Dict[str, Any]:
    """ENAPI"""
    try:
        job_id = str(uuid.uuid4())
        
        # EN
        job = OptimizationJob(job_id=job_id)
        job.total_iterations = max_iterations
        
        optimization_jobs[job_id] = job
        
        # EN
        optimization_thread = threading.Thread(
            target=run_dataset_optimization,
            args=(job,),
            daemon=True
        )
        optimization_thread.start()
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "EN"
        }
        
    except Exception as e:
        logging.error(f"EN: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_optimization_status_api(job_id: str) -> Dict[str, Any]:
    """ENAPI"""
    try:
        if job_id not in optimization_jobs:
            return {
                "success": False,
                "error": "EN"
            }
        
        job = optimization_jobs[job_id]
        
        # EN
        if job.status == "completed":
            logs = job.get_full_logs()
            log_count = len(logs)
        elif job.status == "failed":
            logs = job.get_full_logs()
            log_count = len(logs)
        else:
            logs = job.get_recent_logs(100)
            log_count = len(logs)
        
        # EN
        response_data = {
            "success": True,
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress,
            "current_iteration": job.current_iteration,
            "total_iterations": job.total_iterations,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "duration": job.get_duration(),
            "logs": logs,
            "log_count": log_count,
            "metrics": job.metrics,
            "best_ratio": job.best_ratio,
            "best_f1_score": job.best_f1_score
        }
        
        # EN
        if job.status == "pending":
            response_data["status_description"] = "EN..."
        elif job.status == "running":
            response_data["status_description"] = f"EN... (EN{job.current_iteration}EN)"
        elif job.status == "completed":
            response_data["status_description"] = f"EN!EN: {job.best_ratio:.3f}, F1EN: {job.best_f1_score:.4f}"
        elif job.status == "failed":
            response_data["status_description"] = "EN"
        
        return response_data
        
    except Exception as e:
        logging.error(f"EN: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_optimization_logs_api(job_id: str) -> Dict[str, Any]:
    """ENAPI"""
    try:
        if job_id not in optimization_jobs:
            return {
                "success": False,
                "error": "EN"
            }
        
        job = optimization_jobs[job_id]
        logs = job.get_full_logs()
        
        return {
            "success": True,
            "job_id": job_id,
            "logs": logs,
            "log_count": len(logs)
        }
        
    except Exception as e:
        logging.error(f"EN: {e}")
        return {
            "success": False,
            "error": str(e)
        } 