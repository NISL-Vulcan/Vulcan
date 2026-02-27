#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集优化API接口
为backend_server.py提供数据集优化功能
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

# 全局变量存储优化任务
optimization_jobs = {}

class OptimizationJob:
    """数据集优化任务类"""
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
        # 日志文件路径
        self.log_file_path = f"logs/optimization_{job_id}.log"
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """确保日志目录存在"""
        log_dir = os.path.dirname(self.log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def add_log(self, message: str):
        """添加日志到内存和文件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # 添加到内存中的日志列表
        self.logs.append(log_entry)
        
        # 持久化到文件
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            logging.error(f"写入优化日志文件失败: {e}")
        
        # 限制内存中的日志数量
        if len(self.logs) > 2000:
            self.logs = self.logs[-1500:]
    
    def get_logs(self, limit: int = None) -> List[str]:
        """获取日志，优先从文件读取完整日志"""
        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    file_logs = f.readlines()
                    file_logs = [log.strip() for log in file_logs if log.strip()]
                    
                    if limit:
                        return file_logs[-limit:]
                    return file_logs
        except Exception as e:
            logging.error(f"读取优化日志文件失败: {e}")
        
        if limit:
            return self.logs[-limit:]
        return self.logs.copy()
    
    def get_recent_logs(self, count: int = 50) -> List[str]:
        """获取最近的日志"""
        return self.get_logs(limit=count)
    
    def get_full_logs(self) -> List[str]:
        """获取完整的日志历史"""
        return self.get_logs()
    
    def get_duration(self) -> str:
        """获取任务持续时间"""
        if not self.start_time:
            return "未开始"
        
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
            return f"{hours}小时{minutes}分钟{seconds}秒"
        elif minutes > 0:
            return f"{minutes}分钟{seconds}秒"
        else:
            return f"{seconds}秒"
    
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return self.log_file_path

def run_dataset_optimization(job: OptimizationJob):
    """运行数据集优化任务"""
    try:
        job.status = "running"
        job.start_time = datetime.now().isoformat()
        
        # 在终端和日志中显示详细的启动信息
        startup_msg = f"\n{'='*80}\n🚀 vulcan 数据集优化任务启动\n{'='*80}"
        print(startup_msg)
        job.add_log(startup_msg)
        
        startup_info = f"📋 任务详情:\n   • 任务ID: {job.job_id}\n   • 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(startup_info)
        job.add_log(startup_info)
        
        # 获取当前工作目录和项目根目录
        current_dir = os.path.dirname(__file__)
        project_root = current_dir
        
        print(f"\n📂 环境检查:")
        print(f"   • 当前目录: {current_dir}")
        print(f"   • 项目根目录: {project_root}")
        print(f"   • Python版本: {sys.version.split()[0]}")
        print(f"   • Python路径: {sys.executable}")
        job.add_log(f"📂 工作目录: {project_root}")
        
        # 构建优化脚本路径
        optimization_script = os.path.join(project_root, "auto_update_and_dynamic_ratio.py")
        
        # 详细的文件存在性检查
        print(f"\n🔍 文件检查:")
        
        if not os.path.exists(optimization_script):
            error_msg = f"❌ 数据集优化脚本不存在: {optimization_script}"
            print(error_msg)
            job.add_log(error_msg)
            job.status = "failed"
            return
        else:
            print(f"   ✅ 优化脚本: {optimization_script}")
            job.add_log(f"✅ 找到优化脚本: {optimization_script}")
        
        # 检查数据集文件
        dataset_dir = os.path.join(project_root, "dataset")
        print(f"\n📊 数据集检查:")
        
        if os.path.exists(dataset_dir):
            dataset_files = ["vulnerables.jsonl", "non-vulnerables.jsonl"]
            for dataset_file in dataset_files:
                file_path = os.path.join(dataset_dir, dataset_file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"   ✅ {dataset_file}: {file_size:,} bytes")
                    job.add_log(f"✅ 数据集文件: {dataset_file} ({file_size:,} bytes)")
                else:
                    print(f"   ⚠️  {dataset_file}: 未找到")
                    job.add_log(f"⚠️ 数据集文件缺失: {dataset_file}")
        else:
            print(f"   ⚠️ 数据集目录不存在: {dataset_dir}")
            job.add_log(f"⚠️ 数据集目录不存在: {dataset_dir}")
        
        # 检查GPU
        print(f"\n🎮 硬件环境:")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"   ✅ GPU: {gpu_name} ({gpu_count}个)")
                print(f"   ✅ GPU内存: {gpu_memory:.1f} GB")
                job.add_log(f"✅ GPU: {gpu_name} ({gpu_count}个, {gpu_memory:.1f}GB)")
            else:
                print(f"   ⚠️  GPU: 不可用，将使用CPU")
                job.add_log(f"⚠️ GPU不可用，将使用CPU")
        except ImportError:
            print(f"   ⚠️  PyTorch未安装，无法检测GPU")
            job.add_log(f"⚠️ PyTorch未安装，无法检测GPU")
        
        # 构建优化命令
        cmd = [sys.executable, optimization_script]
        
        print(f"\n🚀 启动数据集优化...")
        print(f"   命令: {' '.join(cmd)}")
        job.add_log(f"🚀 启动数据集优化命令: {' '.join(cmd)}")
        
        # 启动优化进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=project_root
        )
        
        job.process = process
        
        # 实时读取输出并解析
        patterns = {
            'iteration': r'第(\d+)次迭代',  # 匹配 "第1次迭代"
            'ratio': r'比例\s+([0-9.]+)',   # 匹配 "比例 0.5"
            'f1_score': r'F1分数:\s+([0-9.]+)',  # 匹配 "F1分数: 0.8234"
            'best_ratio': r'最佳比例:\s+([0-9.]+)',  # 匹配 "最佳比例: 0.6"
            'best_f1': r'F1分数:\s+([0-9.]+)',  # 匹配 "F1分数: 0.8234"
            'progress': r'进度:\s*(\d+)%',  # 匹配 "进度: 50%"
            'completed': r'搜索完成|最终最佳数据集比例',  # 匹配完成信号
            'failed': r'失败|错误|Error|Exception|程序执行出错'  # 匹配错误信号
        }
        
        current_ratio = 0.0  # 初始化当前比例
        current_f1 = 0.0     # 初始化当前F1分数
        
        for line in process.stdout:
            print(line, end='')  # 实时输出到终端
            job.add_log(line.strip())  # 添加到日志
            
            # 解析进度信息
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    if key == 'iteration':
                        job.current_iteration = int(match.group(1))
                        job.progress = min(100, int((job.current_iteration / job.total_iterations) * 100))
                        print(f"📊 更新进度: {job.progress}% (第{job.current_iteration}次迭代)")
                        job.add_log(f"📊 进度更新: {job.progress}% (第{job.current_iteration}次迭代)")
                    elif key == 'ratio':
                        current_ratio = float(match.group(1))
                        print(f"📊 当前比例: {current_ratio}")
                        job.add_log(f"📊 当前比例: {current_ratio}")
                    elif key == 'f1_score':
                        current_f1 = float(match.group(1))
                        print(f"📊 当前F1分数: {current_f1}")
                        job.add_log(f"📊 当前F1分数: {current_f1}")
                        if current_f1 > job.best_f1_score:
                            job.best_f1_score = current_f1
                            job.best_ratio = current_ratio
                            print(f"🏆 新的最佳记录: 比例={job.best_ratio:.3f}, F1={job.best_f1_score:.4f}")
                            job.add_log(f"🏆 新的最佳记录: 比例={job.best_ratio:.3f}, F1={job.best_f1_score:.4f}")
                    elif key == 'best_ratio':
                        job.best_ratio = float(match.group(1))
                        print(f"🏆 最佳比例: {job.best_ratio}")
                        job.add_log(f"🏆 最佳比例: {job.best_ratio}")
                    elif key == 'progress':
                        # 直接使用脚本输出的进度百分比
                        progress_percent = int(match.group(1))
                        job.progress = progress_percent
                        print(f"📊 脚本进度: {progress_percent}%")
                        job.add_log(f"📊 脚本进度: {progress_percent}%")
                    elif key == 'completed':
                        job.status = "completed"
                        job.progress = 100
                        print("✅ 检测到完成信号")
                        job.add_log("✅ 检测到完成信号")
                    elif key == 'failed':
                        job.status = "failed"
                        print("❌ 检测到错误信号")
                        job.add_log("❌ 检测到错误信号")
        
        # 等待进程完成
        return_code = process.wait()
        job.end_time = datetime.now().isoformat()
        
        if return_code == 0 and job.status != "failed":
            job.status = "completed"
            job.progress = 100
            
            completion_msg = f"\n✅ 数据集优化任务完成！"
            completion_msg += f"\n📊 优化结果:"
            completion_msg += f"\n   • 最佳数据集比例: {job.best_ratio:.3f}"
            completion_msg += f"\n   • 最佳F1分数: {job.best_f1_score:.4f}"
            completion_msg += f"\n   • 总迭代次数: {job.current_iteration}"
            completion_msg += f"\n   • 总耗时: {job.get_duration()}"
            
            print(completion_msg)
            job.add_log(completion_msg)
            
            # 保存最终指标
            job.metrics = {
                "best_ratio": job.best_ratio,
                "best_f1_score": job.best_f1_score,
                "total_iterations": job.current_iteration,
                "duration": job.get_duration()
            }
        else:
            job.status = "failed"
            error_msg = f"\n❌ 数据集优化任务失败！返回码: {return_code}"
            print(error_msg)
            job.add_log(error_msg)
            
    except Exception as e:
        job.status = "failed"
        job.end_time = datetime.now().isoformat()
        error_msg = f"数据集优化任务异常: {str(e)}"
        print(error_msg)
        job.add_log(error_msg)
        logging.error(f"数据集优化任务异常: {e}")
        import traceback
        job.add_log(traceback.format_exc())

def start_dataset_optimization_api(max_iterations: int = 15) -> Dict[str, Any]:
    """启动数据集优化API"""
    try:
        job_id = str(uuid.uuid4())
        
        # 创建优化任务
        job = OptimizationJob(job_id=job_id)
        job.total_iterations = max_iterations
        
        optimization_jobs[job_id] = job
        
        # 启动优化线程
        optimization_thread = threading.Thread(
            target=run_dataset_optimization,
            args=(job,),
            daemon=True
        )
        optimization_thread.start()
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "数据集优化任务已启动"
        }
        
    except Exception as e:
        logging.error(f"启动数据集优化失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_optimization_status_api(job_id: str) -> Dict[str, Any]:
    """获取数据集优化状态API"""
    try:
        if job_id not in optimization_jobs:
            return {
                "success": False,
                "error": "优化任务不存在"
            }
        
        job = optimization_jobs[job_id]
        
        # 根据任务状态决定返回的日志数量
        if job.status == "completed":
            logs = job.get_full_logs()
            log_count = len(logs)
        elif job.status == "failed":
            logs = job.get_full_logs()
            log_count = len(logs)
        else:
            logs = job.get_recent_logs(100)
            log_count = len(logs)
        
        # 构建响应数据
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
        
        # 根据状态添加描述
        if job.status == "pending":
            response_data["status_description"] = "等待启动数据集优化任务..."
        elif job.status == "running":
            response_data["status_description"] = f"正在优化数据集比例... (第{job.current_iteration}次迭代)"
        elif job.status == "completed":
            response_data["status_description"] = f"数据集优化完成！最佳比例: {job.best_ratio:.3f}, F1分数: {job.best_f1_score:.4f}"
        elif job.status == "failed":
            response_data["status_description"] = "数据集优化任务失败"
        
        return response_data
        
    except Exception as e:
        logging.error(f"获取优化状态失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_optimization_logs_api(job_id: str) -> Dict[str, Any]:
    """获取数据集优化日志API"""
    try:
        if job_id not in optimization_jobs:
            return {
                "success": False,
                "error": "优化任务不存在"
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
        logging.error(f"获取优化日志失败: {e}")
        return {
            "success": False,
            "error": str(e)
        } 