#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vulcan-Detection Backend API Server
提供配置文件生成和训练接口
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
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
import traceback
import time
import re
from typing import Dict, Any, List, Optional
import requests

# 添加framework路径以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'framework'))

# 导入配置模板系统
from vulcan.framework.config_templates import ConfigTemplateManager

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储训练任务
training_jobs = {}
training_configs = {}
optimization_jobs = {}  # 新增：存储数据集优化任务
jobs = {}  # 新增：存储所有类型的任务（包括论文检索任务）

class TrainingJob:
    """训练任务类"""
    def __init__(self, job_id: str, config_path: str, model_name: str):
        self.job_id = job_id
        self.config_path = config_path
        self.model_name = model_name
        self.status = "pending"  # pending, running, completed, failed
        self.status_description = "等待开始训练..."  # 添加状态描述属性
        self.progress = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_iteration = 0  # 添加当前迭代属性
        self.total_iterations = 0   # 添加总迭代属性
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.process = None
        self.metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
        # 日志文件路径
        self.log_file_path = f"logs/training_{job_id}.log"
        # 验证状态标志 - 防止重复验证
        self._validation_started = False
        # 自动验证标志
        self.auto_validation = False  # 默认不启用自动验证，需要明确指定
        # 当前阶段标识
        self.current_phase = "training"  # training, validation, completed
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """确保日志目录存在"""
        import os
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
            logger.error(f"写入日志文件失败: {e}")
        
        # 限制内存中的日志数量，但保留更多历史记录
        if len(self.logs) > 2000:  # 增加内存中的日志数量限制
            self.logs = self.logs[-1500:]  # 保留更多历史记录
    
    def get_logs(self, limit: int = None) -> List[str]:
        """获取日志，优先从文件读取完整日志"""
        try:
            # 尝试从文件读取完整日志
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    file_logs = f.readlines()
                    # 去除换行符并过滤空行
                    file_logs = [log.strip() for log in file_logs if log.strip()]
                    
                    if limit:
                        return file_logs[-limit:]
                    return file_logs
        except Exception as e:
            logger.error(f"读取日志文件失败: {e}")
        
        # 如果文件读取失败，返回内存中的日志
        if limit:
            return self.logs[-limit:]
        return self.logs.copy()
    
    def get_recent_logs(self, count: int = 50) -> List[str]:
        """获取最近的日志"""
        return self.get_logs(limit=count)
    
    def get_full_logs(self) -> List[str]:
        """获取完整的日志历史"""
        return self.get_logs()
    
    def save_logs_to_file(self):
        """将内存中的日志保存到文件"""
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                for log in self.logs:
                    f.write(log + '\n')
        except Exception as e:
            logger.error(f"保存日志到文件失败: {e}")
    
    def load_logs_from_file(self):
        """从文件加载日志到内存"""
        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    file_logs = f.readlines()
                    self.logs = [log.strip() for log in file_logs if log.strip()]
        except Exception as e:
            logger.error(f"从文件加载日志失败: {e}")

    def get_duration(self) -> str:
        """计算任务持续时间"""
        if self.start_time and self.end_time:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(self.end_time)
            duration = end_dt - start_dt
            return str(duration)
        return "N/A"
    
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return self.log_file_path

class ConfigGenerator:
    """配置文件生成器"""
    
    def __init__(self, configs_dir: str = "generated_configs"):
        self.configs_dir = Path(configs_dir)
        self.configs_dir.mkdir(exist_ok=True)  # 确保目录存在
        self.template_manager = ConfigTemplateManager()
        self.legacy_templates = self._load_legacy_templates()
    
    def _load_legacy_templates(self) -> Dict[str, Dict]:
        """加载现有的配置模板作为备用"""
        templates = {}
        # 从configs目录加载现有配置文件
        config_root = Path("configs")
        if config_root.exists():
            for config_file in config_root.glob("*.yaml"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        templates[config_file.stem] = config
                except Exception as e:
                    logger.error(f"加载配置模板失败 {config_file}: {e}")
        return templates
    
    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        # 从模板管理器获取模型列表
        template_models = self.template_manager.list_models()
        
        # 从legacy模板获取模型列表
        legacy_models = set()
        for template in self.legacy_templates.values():
            if 'MODEL' in template and 'NAME' in template['MODEL']:
                legacy_models.add(template['MODEL']['NAME'])
        
        # 合并并去重
        all_models = list(set(template_models + list(legacy_models)))
        return all_models
    
    def get_available_datasets(self) -> List[str]:
        """获取可用的数据集列表"""
        datasets = set()
        
        # 从模板管理器获取数据集
        for template_name in self.template_manager.list_templates():
            template = self.template_manager.get_template(template_name)
            if template and template.template.get('DATASET', {}).get('NAME'):
                datasets.add(template.template['DATASET']['NAME'])
        
        # 从legacy模板获取
        for template in self.legacy_templates.values():
            if 'DATASET' in template and 'NAME' in template['DATASET']:
                datasets.add(template['DATASET']['NAME'])
        
        return list(datasets)
    
    def generate_config(self, 
                       model_name: str, 
                       dataset_name: str, 
                       training_params: Dict[str, Any],
                       device: str = "cuda") -> Dict[str, Any]:
        """生成配置文件"""
        try:
            # 验证是否提供了必需的训练参数
            required_params = ['epochs', 'batch_size', 'learning_rate']
            missing_params = [param for param in required_params if param not in training_params or training_params[param] is None]
            
            if missing_params:
                raise ValueError(f"缺少必需的训练参数: {', '.join(missing_params)}")
            
            # 验证参数值的有效性
            if training_params['epochs'] <= 0:
                raise ValueError("epochs 必须大于 0")
            if training_params['batch_size'] <= 0:
                raise ValueError("batch_size 必须大于 0")
            if training_params['learning_rate'] <= 0:
                raise ValueError("learning_rate 必须大于 0")
            
            # 首先尝试使用新的模板系统
            template = self.template_manager.get_template_by_model(model_name)
            if template:
                logger.info(f"使用模板系统生成 {model_name} 配置")
                
                # 构建参数映射，使用真实值
                config_params = {
                    'DEVICE': device,
                    'TRAIN.BATCH_SIZE': training_params['batch_size'],
                    'TRAIN.EPOCHS': training_params['epochs'],
                    'TRAIN.EVAL_INTERVAL': training_params.get('eval_interval', 1),
                    'OPTIMIZER.LR': training_params['learning_rate'],
                    'SAVE_DIR': training_params.get('save_dir', 'output'),
                    'DATASET.NAME': dataset_name
                }
                
                # 根据数据集名称设置数据路径
                if training_params.get('data_path'):
                    config_params['DATASET.ROOT'] = training_params['data_path']
                
                config = template.generate_config(**config_params)
                return config
            
            # 如果模板系统失败，使用legacy方法
            logger.info(f"使用legacy方法生成 {model_name} 配置")
            best_template = self._find_best_legacy_template(model_name, dataset_name)
            
            if best_template:
                config = self._deep_copy_dict(best_template)
                
                # 更新配置参数，使用真实值
                config['DEVICE'] = device
                config['SAVE_DIR'] = training_params.get('save_dir', 'output')
                
                if 'TRAIN' not in config:
                    config['TRAIN'] = {}
                config['TRAIN']['BATCH_SIZE'] = training_params['batch_size']
                config['TRAIN']['EPOCHS'] = training_params['epochs']
                config['TRAIN']['EVAL_INTERVAL'] = training_params.get('eval_interval', 1)
                
                if 'OPTIMIZER' not in config:
                    config['OPTIMIZER'] = {}
                config['OPTIMIZER']['LR'] = training_params['learning_rate']
                
                if 'DATASET' not in config:
                    config['DATASET'] = {}
                config['DATASET']['NAME'] = dataset_name
                
                if training_params.get('data_path'):
                    config['DATASET']['ROOT'] = training_params['data_path']
                
                return config
            
            # 如果都失败了，生成基础配置
            logger.warning(f"未找到 {model_name} 的模板，生成基础配置")
            return self._generate_basic_config(model_name, dataset_name, training_params, device)
            
        except Exception as e:
            logger.error(f"生成配置失败: {e}")
            # 返回基础配置作为后备
            return self._generate_basic_config(model_name, dataset_name, training_params, device)
    
    def _generate_basic_config(self, model_name: str, dataset_name: str, 
                              training_params: Dict[str, Any], device: str) -> Dict[str, Any]:
        """生成基础配置"""
        # 验证必需参数
        if 'epochs' not in training_params or training_params['epochs'] <= 0:
            raise ValueError("epochs 是必需的且必须大于 0")
        if 'batch_size' not in training_params or training_params['batch_size'] <= 0:
            raise ValueError("batch_size 是必需的且必须大于 0")
        if 'learning_rate' not in training_params or training_params['learning_rate'] <= 0:
            raise ValueError("learning_rate 是必需的且必须大于 0")
        
        return {
            'DEVICE': device,
            'SAVE_DIR': training_params.get('save_dir', 'output'),
            'MODEL': {
                'NAME': model_name,
                'BACKBONE': '',
                'PARAMS': {},
                'PRETRAINED': None
            },
            'DATASET': {
                'NAME': dataset_name,
                'ROOT': training_params.get('data_path', ''),
                'PARAMS': {}
            },
            'TRAIN': {
                'BATCH_SIZE': training_params['batch_size'],
                'EPOCHS': training_params['epochs'],
                'EVAL_INTERVAL': training_params.get('eval_interval', 1),
                'AMP': False,
                'DDP': False
            },
            'EVAL': {
                'MODEL_PATH': ''
            },
            'LOSS': {
                'NAME': 'cross_entropy'
            },
            'OPTIMIZER': {
                'NAME': 'adamw',
                'LR': training_params['learning_rate'],
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'polynomial',
                'POWER': 0.9,
                'WARMUP': 10,
                'WARMUP_RATIO': 0.1
            }
        }
    
    def _find_best_legacy_template(self, model_name: str, dataset_name: str) -> Optional[Dict]:
        """查找最佳匹配的legacy模板"""
        best_match = None
        best_score = 0
        
        for template_name, template in self.legacy_templates.items():
            score = 0
            
            # 检查模型匹配度
            if 'MODEL' in template and 'NAME' in template['MODEL']:
                template_model = template['MODEL']['NAME'].lower()
                if model_name.lower() in template_model or template_model in model_name.lower():
                    score += 10
            
            # 检查数据集匹配度
            if 'DATASET' in template and 'NAME' in template['DATASET']:
                template_dataset = template['DATASET']['NAME'].lower()
                if dataset_name.lower() in template_dataset or template_dataset in dataset_name.lower():
                    score += 5
            
            if score > best_score:
                best_score = score
                best_match = template
        
        # 如果没有找到匹配的，返回第一个模板
        if best_match is None and self.legacy_templates:
            best_match = list(self.legacy_templates.values())[0]
        
        return best_match
    
    def _deep_copy_dict(self, d: Dict) -> Dict:
        """深度复制字典"""
        import copy
        return copy.deepcopy(d)
    
    def save_config(self, config: Dict[str, Any], filename: str) -> str:
        """保存配置文件"""
        output_dir = Path("generated_configs")
        output_dir.mkdir(exist_ok=True)
        
        config_path = output_dir / f"{filename}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        return str(config_path)

# 初始化配置生成器
config_generator = ConfigGenerator()

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用的模型列表"""
    try:
        models = config_generator.get_available_models()
        return jsonify({
            "success": True,
            "models": models
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """获取可用的数据集列表"""
    try:
        datasets = config_generator.get_available_datasets()
        return jsonify({
            "success": True,
            "datasets": datasets
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/generate-config', methods=['POST'])
def generate_config():
    """生成配置文件"""
    try:
        data = request.json
        
        # 获取参数
        model_name = data.get('model_name', 'DeepWuKong')
        dataset_name = data.get('dataset_name', 'DWK_Dataset')
        device = data.get('device', 'cuda')
        
        training_params = {
            'batch_size': data.get('batch_size', 32),
            'epochs': data.get('epochs', 10),
            'learning_rate': data.get('learning_rate', 0.001),
            'eval_interval': data.get('eval_interval', 1),
            'save_dir': data.get('save_dir', 'output'),
            'data_path': data.get('data_path', '')
        }
        
        # 生成配置
        config = config_generator.generate_config(
            model_name=model_name,
            dataset_name=dataset_name,
            training_params=training_params,
            device=device
        )
        
        # 生成配置ID
        config_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset_name}_{timestamp}"
        
        # 保存配置
        config_path = config_generator.save_config(config, filename)
        
        # 存储配置信息
        training_configs[config_id] = {
            'config': config,
            'config_path': config_path,
            'filename': filename,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'created_time': datetime.now().isoformat()
        }
        
        return jsonify({
            "success": True,
            "config_id": config_id,
            "config": config,
            "config_path": config_path,
            "message": f"配置文件已生成: {filename}.yaml"
        })
        
    except Exception as e:
        logger.error(f"生成配置文件失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/generate-config-from-template', methods=['POST'])
def generate_config_from_template():
    """基于 regvd_reveal.yaml 模板生成配置文件"""
    try:
        data = request.json
        
        # 获取参数，使用适合 regvd_reveal 模板的默认值
        model_name = data.get('model_name', 'GNNReGVD')
        dataset_name = data.get('dataset_name', 'ReGVD')
        device = data.get('device', 'cuda')
        batch_size = data.get('batch_size', 128)
        epochs = data.get('epochs', 2)
        learning_rate = data.get('learning_rate', 0.001)
        save_dir = data.get('save_dir', 'output')
        
        # 读取 regvd_reveal.yaml 模板
        template_path = os.path.join('configs', 'regvd_reveal.yaml')
        if not os.path.exists(template_path):
            return jsonify({
                "success": False,
                "error": f"模板文件 regvd_reveal.yaml 不存在于 configs 目录"
            }), 404
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # 应用参数替换
        config_content = template_content.replace('cuda', device)
        config_content = config_content.replace('output', save_dir)
        config_content = config_content.replace('GNNReGVD', model_name)
        config_content = config_content.replace('ReGVD', dataset_name)
        config_content = config_content.replace('128', str(batch_size))
        config_content = config_content.replace('2', str(epochs))
        config_content = config_content.replace('0.001', str(learning_rate))
        
        # 生成有辨识度的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = f"regvd_reveal_{model_name}_{timestamp}"
        
        # 保存到 configs 目录
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)
        config_path = configs_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # 解析生成的配置用于返回
        try:
            config_data = yaml.safe_load(config_content)
        except:
            config_data = {}
        
        logger.info(f"基于模板生成配置文件: {config_path}")
        
        return jsonify({
            "success": True,
            "config_name": config_name,
            "config_path": str(config_path),
            "config_content": config_content,
            "config_data": config_data,
            "template_used": "regvd_reveal",
            "message": f"基于 regvd_reveal.yaml 模板生成配置文件: {config_name}.yaml"
        })
        
    except Exception as e:
        logger.error(f"基于模板生成配置文件失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/start-training', methods=['POST'])
def start_training():
    """启动训练"""
    try:
        data = request.json
        config_id = data.get('config_id')
        auto_validation = data.get('auto_validation', False)  # 默认不自动验证
        
        if not config_id or config_id not in training_configs:
            return jsonify({
                "success": False,
                "error": "无效的配置ID"
            }), 400
        
        config_info = training_configs[config_id]
        job_id = str(uuid.uuid4())
        
        # 创建训练任务
        job = TrainingJob(
            job_id=job_id,
            config_path=config_info['config_path'],
            model_name=config_info['model_name']
        )
        
        # 设置自动验证标志
        job.auto_validation = auto_validation
        
        # 获取训练参数
        config = config_info['config']
        job.total_epochs = config.get('TRAIN', {}).get('EPOCHS', 10)
        
        training_jobs[job_id] = job
        
        # 启动训练线程
        training_thread = threading.Thread(
            target=run_training,
            args=(job,),
            daemon=True
        )
        training_thread.start()
        
        validation_message = "，完成后将自动验证" if auto_validation else ""
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "auto_validation": auto_validation,
            "message": f"训练任务已启动: {job.model_name}{validation_message}"
        })
        
    except Exception as e:
        logger.error(f"启动训练失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/training-status/<job_id>', methods=['GET'])
def get_training_status(job_id: str):
    """获取训练状态"""
    try:
        print(f"🔍 [API] 查询训练状态: {job_id}")
        
        if job_id not in training_jobs:
            error_msg = "训练任务不存在"
            print(f"❌ {error_msg}: {job_id}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        job = training_jobs[job_id]
        
        print(f"📊 训练状态: {job.status}, 进度: {job.progress}%, 当前轮次: {job.current_epoch}/{job.total_epochs}")
        
        # 检测当前是否在验证阶段
        is_validation_phase = job.metrics and (
            job.metrics.get('validation_accuracy', 0) > 0 or
            job.metrics.get('validation_f1', 0) > 0 or
            job.metrics.get('validation_precision', 0) > 0 or
            job.metrics.get('validation_recall', 0) > 0 or
            job.metrics.get('validation_loss', 0) > 0 or
            job.metrics.get('validation_auc', 0) > 0
        )
        
        # 分离训练指标和验证指标
        training_metrics = {}
        validation_metrics = {}
        
        for key, value in job.metrics.items():
            if key.startswith('validation_'):
                validation_metrics[key] = value
            else:
                training_metrics[key] = value
        
        if training_metrics:
            print(f"🎯 当前训练指标: {training_metrics}")
        if validation_metrics:
            print(f"🔬 当前验证指标: {validation_metrics}")
        
        # 根据任务状态决定返回的日志数量
        if job.status == "completed":
            # 任务完成后，返回完整的日志历史
            logs = job.get_full_logs()
            log_count = len(logs)
            print(f"✅ 训练已完成，返回完整日志({log_count}条)")
        elif job.status == "failed":
            # 任务失败后，也返回完整日志用于调试
            logs = job.get_full_logs()
            log_count = len(logs)
            print(f"❌ 训练失败，返回完整日志({log_count}条)")
        else:
            # 任务进行中，返回最近的日志
            logs = job.get_recent_logs(100)  # 增加返回的日志数量
            log_count = len(logs)
            print(f"🔄 训练进行中，返回最近日志({log_count}条)")
        
        # 构建响应数据
        response_data = {
            "success": True,
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "current_iteration": job.current_iteration,  # 添加当前迭代
            "total_iterations": job.total_iterations,    # 添加总迭代数
            "start_time": job.start_time,
            "end_time": job.end_time,
            "logs": logs,
            "log_count": log_count,  # 添加日志数量信息
            "log_file_path": job.get_log_file_path(),  # 添加日志文件路径
            
            # 训练相关指标
            "training_metrics": training_metrics,
            
            # 验证相关指标
            "validation_metrics": validation_metrics,
            "is_validation_phase": is_validation_phase,
            
            # 兼容性：保持原有的 metrics 字段
            "metrics": job.metrics,
            
            # 任务摘要信息
            "model_name": job.model_name,
            "config_path": job.config_path,
            
            # 状态描述
            "status_description": get_status_description(job.status, is_validation_phase),
            
            # 性能摘要
            "performance_summary": generate_performance_summary(training_metrics, validation_metrics),
            
            # 新增：训练阶段信息
            "auto_validation": getattr(job, 'auto_validation', True),
            "current_phase": getattr(job, 'current_phase', 'training'),
            "training_completed": job.current_phase in ['validation', 'completed'] if hasattr(job, 'current_phase') else (job.status == 'completed'),
            "validation_completed": job.current_phase == 'completed' if hasattr(job, 'current_phase') else bool(validation_metrics),
            "phases": {
                "training": {
                    "status": "completed" if (hasattr(job, 'current_phase') and job.current_phase in ['validation', 'completed']) else job.status,
                    "progress": 100 if (hasattr(job, 'current_phase') and job.current_phase in ['validation', 'completed']) else job.progress
                },
                "validation": {
                    "status": "completed" if (hasattr(job, 'current_phase') and job.current_phase == 'completed') else 
                             "running" if (hasattr(job, 'current_phase') and job.current_phase == 'validation') else
                             "pending" if (hasattr(job, 'current_phase') and job.current_phase == 'training') else
                             "completed" if validation_metrics else "pending",
                    "enabled": getattr(job, 'auto_validation', True)
                }
            }
        }
        
        # 如果任务已完成，添加最终结果摘要
        if job.status == "completed":
            response_data["final_results"] = generate_final_results_summary(training_metrics, validation_metrics, job)
            # 添加完成后的日志摘要
            response_data["completion_summary"] = {
                "total_logs": log_count,
                "training_duration": job.get_duration(),
                "final_logs_preview": logs[-20:] if len(logs) > 20 else logs,  # 最后20条日志预览
                "log_file_available": os.path.exists(job.get_log_file_path())
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"获取训练状态失败: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def get_status_description(status: str, is_validation_phase: bool) -> str:
    """获取状态描述"""
    if status == "pending":
        return "等待中"
    elif status == "running":
        if is_validation_phase:
            return "验证中"
        else:
            return "训练中"
    elif status == "completed":
        return "已完成"
    elif status == "failed":
        return "失败"
    else:
        return "未知状态"

def generate_performance_summary(training_metrics: dict, validation_metrics: dict) -> dict:
    """生成性能摘要"""
    summary = {
        "training": {},
        "validation": {},
        "comparison": {}
    }
    
    # 训练性能摘要
    if training_metrics:
        if "accuracy" in training_metrics:
            summary["training"]["accuracy"] = {
                "value": training_metrics["accuracy"],
                "percentage": training_metrics["accuracy"] * 100,
                "formatted": f"{training_metrics['accuracy']:.4f} ({training_metrics['accuracy']*100:.1f}%)"
            }
        if "loss" in training_metrics:
            summary["training"]["loss"] = {
                "value": training_metrics["loss"],
                "formatted": f"{training_metrics['loss']:.4f}"
            }
        if "f1" in training_metrics:
            summary["training"]["f1"] = {
                "value": training_metrics["f1"],
                "percentage": training_metrics["f1"] * 100,
                "formatted": f"{training_metrics['f1']:.4f} ({training_metrics['f1']*100:.1f}%)"
            }
    
    # 验证性能摘要
    if validation_metrics:
        if "validation_accuracy" in validation_metrics:
            summary["validation"]["accuracy"] = {
                "value": validation_metrics["validation_accuracy"],
                "percentage": validation_metrics["validation_accuracy"] * 100,
                "formatted": f"{validation_metrics['validation_accuracy']:.4f} ({validation_metrics['validation_accuracy']*100:.1f}%)"
            }
        if "validation_loss" in validation_metrics:
            summary["validation"]["loss"] = {
                "value": validation_metrics["validation_loss"],
                "formatted": f"{validation_metrics['validation_loss']:.4f}"
            }
        if "validation_f1" in validation_metrics:
            summary["validation"]["f1"] = {
                "value": validation_metrics["validation_f1"],
                "percentage": validation_metrics["validation_f1"] * 100,
                "formatted": f"{validation_metrics['validation_f1']:.4f} ({validation_metrics['validation_f1']*100:.1f}%)"
            }
        if "validation_precision" in validation_metrics:
            summary["validation"]["precision"] = {
                "value": validation_metrics["validation_precision"],
                "percentage": validation_metrics["validation_precision"] * 100,
                "formatted": f"{validation_metrics['validation_precision']:.4f} ({validation_metrics['validation_precision']*100:.1f}%)"
            }
        if "validation_recall" in validation_metrics:
            summary["validation"]["recall"] = {
                "value": validation_metrics["validation_recall"],
                "percentage": validation_metrics["validation_recall"] * 100,
                "formatted": f"{validation_metrics['validation_recall']:.4f} ({validation_metrics['validation_recall']*100:.1f}%)"
            }
        if "validation_auc" in validation_metrics:
            summary["validation"]["auc"] = {
                "value": validation_metrics["validation_auc"],
                "formatted": f"{validation_metrics['validation_auc']:.4f}"
            }
        if "validation_overall_score" in validation_metrics:
            summary["validation"]["overall_score"] = {
                "value": validation_metrics["validation_overall_score"],
                "percentage": validation_metrics["validation_overall_score"] * 100,
                "formatted": f"{validation_metrics['validation_overall_score']:.4f} ({validation_metrics['validation_overall_score']*100:.1f}%)"
            }
    
    # 训练vs验证对比
    if "accuracy" in training_metrics and "validation_accuracy" in validation_metrics:
        diff = validation_metrics["validation_accuracy"] - training_metrics["accuracy"]
        summary["comparison"]["accuracy_diff"] = {
            "value": diff,
            "percentage": diff * 100,
            "formatted": f"{diff:.4f} ({diff*100:+.1f}%)",
            "interpretation": "更好" if diff > 0 else "过拟合" if diff < -0.05 else "正常"
        }
    
    return summary

def generate_final_results_summary(training_metrics: dict, validation_metrics: dict, job) -> dict:
    """生成最终结果摘要"""
    final_results = {
        "completion_time": job.end_time,
        "duration": job.get_duration(),
        "model_name": job.model_name,
        "config_path": job.config_path,
        "training_completed": True,
        "validation_completed": bool(validation_metrics),
        "key_metrics": {},
        "recommendations": []
    }
    
    # 关键指标
    if validation_metrics:
        # 使用验证指标作为最终评判标准
        if "validation_accuracy" in validation_metrics:
            final_results["key_metrics"]["final_accuracy"] = {
                "value": validation_metrics["validation_accuracy"],
                "formatted": f"{validation_metrics['validation_accuracy']:.4f} ({validation_metrics['validation_accuracy']*100:.1f}%)",
                "source": "validation"
            }
        
        if "validation_f1" in validation_metrics:
            final_results["key_metrics"]["final_f1"] = {
                "value": validation_metrics["validation_f1"],
                "formatted": f"{validation_metrics['validation_f1']:.4f} ({validation_metrics['validation_f1']*100:.1f}%)",
                "source": "validation"
            }
        
        if "validation_overall_score" in validation_metrics:
            final_results["key_metrics"]["overall_score"] = {
                "value": validation_metrics["validation_overall_score"],
                "formatted": f"{validation_metrics['validation_overall_score']:.4f} ({validation_metrics['validation_overall_score']*100:.1f}%)",
                "source": "validation"
            }
    elif training_metrics:
        # 如果没有验证指标，使用训练指标
        if "accuracy" in training_metrics:
            final_results["key_metrics"]["final_accuracy"] = {
                "value": training_metrics["accuracy"],
                "formatted": f"{training_metrics['accuracy']:.4f} ({training_metrics['accuracy']*100:.1f}%)",
                "source": "training"
            }
    
    # 生成建议
    if validation_metrics:
        acc = validation_metrics.get("validation_accuracy", 0)
        f1 = validation_metrics.get("validation_f1", 0)
        
        if acc > 0.9:
            final_results["recommendations"].append("🎉 模型性能优秀！可以考虑部署使用。")
        elif acc > 0.8:
            final_results["recommendations"].append("👍 模型性能良好，可以进一步优化或直接使用。")
        elif acc > 0.7:
            final_results["recommendations"].append("⚠️ 模型性能一般，建议调整超参数或增加训练数据。")
        else:
            final_results["recommendations"].append("❌ 模型性能较差，建议检查数据质量和模型架构。")
        
        if f1 > 0 and abs(acc - f1) > 0.1:
            final_results["recommendations"].append("📊 准确率和F1分数差异较大，建议检查数据平衡性。")
    
    # 训练vs验证对比建议
    if training_metrics and validation_metrics:
        train_acc = training_metrics.get("accuracy", 0)
        val_acc = validation_metrics.get("validation_accuracy", 0)
        
        if train_acc > 0 and val_acc > 0:
            diff = train_acc - val_acc
            if diff > 0.1:
                final_results["recommendations"].append("🔍 存在过拟合现象，建议使用正则化或增加验证数据。")
            elif diff < -0.05:
                final_results["recommendations"].append("🤔 验证性能超过训练性能，请检查数据划分。")
    
    return final_results

@app.route('/api/training-logs/<job_id>', methods=['GET'])
def get_training_logs(job_id: str):
    """获取训练日志"""
    try:
        if job_id not in training_jobs:
            return jsonify({
                "success": False,
                "error": "训练任务不存在"
            }), 404
        
        job = training_jobs[job_id]
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "logs": job.get_full_logs()
        })
        
    except Exception as e:
        logger.error(f"获取训练日志失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# 添加新的API端点用于获取现有配置文件
@app.route('/api/get-config/<config_name>', methods=['GET'])
def get_existing_config(config_name: str):
    """获取现有的配置文件"""
    try:
        # 构建配置文件路径
        config_path = os.path.join('configs', f'{config_name}.yaml')
        
        if not os.path.exists(config_path):
            return jsonify({
                "success": False,
                "error": f"配置文件 {config_name}.yaml 不存在"
            }), 404
        
        # 读取配置文件内容
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # 重新打开文件读取YAML数据
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.warning(f"YAML解析警告: {e}")
                config_data = {}
        
        return jsonify({
            "success": True,
            "config_name": config_name,
            "config_path": config_path,
            "config_content": config_content,
            "config_data": config_data,
            "message": f"成功获取配置文件: {config_name}.yaml"
        })
        
    except Exception as e:
        logger.error(f"获取配置文件失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/start-training-with-config-id/<config_id>', methods=['POST'])
def start_training_with_config_id(config_id: str):
    """使用配置ID启动训练（适用于generated_configs中的配置）"""
    try:
        # 获取请求体数据
        data = request.json if request.json else {}
        auto_validation = data.get('auto_validation', False)  # 默认不自动验证
        
        print(f"\n🚀 [API] 接收到训练启动请求（配置ID）")
        print(f"🆔 配置ID: {config_id}")
        print(f"🔬 自动验证: {auto_validation}")
        
        # 检查配置ID是否存在
        if config_id not in training_configs:
            error_msg = f"配置ID {config_id} 不存在"
            print(f"❌ {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        config_info = training_configs[config_id]
        config_path = config_info['config_path']
        model_name = config_info['model_name']
        
        print(f"📄 配置文件路径: {config_path}")
        print(f"🤖 训练模型: {model_name}")
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            error_msg = f"配置文件 {config_path} 不存在"
            print(f"❌ {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        # 生成训练任务ID
        job_id = str(uuid.uuid4())
        print(f"🆔 训练任务ID: {job_id}")
        
        # 创建训练任务
        job = TrainingJob(
            job_id=job_id,
            config_path=config_path,
            model_name=model_name
        )
        
        # 设置自动验证标志
        job.auto_validation = auto_validation
        
        # 从配置信息获取训练参数
        if 'config' in config_info and config_info['config']:
            config_data = config_info['config']
            if 'TRAIN' in config_data:
                job.total_epochs = config_data['TRAIN'].get('EPOCHS', 10)
            else:
                job.total_epochs = 10
        else:
            job.total_epochs = 10
        
        print(f"📊 总训练轮数: {job.total_epochs}")
        
        training_jobs[job_id] = job
        
        print(f"🚀 启动训练线程...")
        
        # 启动训练线程
        training_thread = threading.Thread(
            target=run_training,
            args=(job,),
            daemon=True
        )
        training_thread.start()
        
        validation_message = "，完成后将自动验证" if auto_validation else ""
        success_msg = f"训练任务已启动: {model_name} (使用配置ID {config_id}){validation_message}"
        print(f"✅ {success_msg}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "config_id": config_id,
            "config_path": config_path,
            "model_name": model_name,
            "auto_validation": auto_validation,
            "filename": config_info.get('filename', 'unknown'),
            "message": success_msg
        })
        
    except Exception as e:
        error_msg = f"使用配置ID启动训练失败: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/start-training-with-config/<config_name>', methods=['POST'])
def start_training_with_existing_config(config_name: str):
    """使用现有配置文件启动训练"""
    try:
        # 获取请求体数据
        data = request.json if request.json else {}
        auto_validation = data.get('auto_validation', False)  # 默认不自动验证
        
        print(f"\n🚀 [API] 接收到训练启动请求")
        print(f"📋 配置文件: {config_name}.yaml")
        print(f"🔬 自动验证: {auto_validation}")
        
        # 构建配置文件路径
        config_path = os.path.join('configs', f'{config_name}.yaml')
        
        if not os.path.exists(config_path):
            error_msg = f"配置文件 {config_name}.yaml 不存在"
            print(f"❌ {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        # 读取配置文件获取模型名称
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            model_name = config_data.get('MODEL', {}).get('NAME', config_name)
        except:
            model_name = config_name
        
        print(f"🤖 训练模型: {model_name}")
        
        # 生成训练任务ID
        job_id = str(uuid.uuid4())
        print(f"🆔 训练任务ID: {job_id}")
        
        # 创建训练任务
        job = TrainingJob(
            job_id=job_id,
            config_path=config_path,
            model_name=model_name
        )
        
        # 设置自动验证标志
        job.auto_validation = auto_validation
        
        # 从配置文件获取训练参数
        if config_data and 'TRAIN' in config_data:
            job.total_epochs = config_data['TRAIN'].get('EPOCHS', 10)
        else:
            job.total_epochs = 10
        
        print(f"📊 总训练轮数: {job.total_epochs}")
        
        training_jobs[job_id] = job
        
        print(f"🚀 启动训练线程...")
        
        # 启动训练线程
        training_thread = threading.Thread(
            target=run_training,
            args=(job,),
            daemon=True
        )
        training_thread.start()
        
        validation_message = "，完成后将自动验证" if auto_validation else ""
        success_msg = f"训练任务已启动: {model_name} (使用 {config_name}.yaml){validation_message}"
        print(f"✅ {success_msg}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "config_name": config_name,
            "config_path": config_path,
            "model_name": model_name,
            "auto_validation": auto_validation,
            "message": success_msg
        })
        
    except Exception as e:
        error_msg = f"启动训练失败: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/list-configs', methods=['GET'])
def list_existing_configs():
    """列出所有现有的配置文件"""
    try:
        config_dir = Path('configs')
        if not config_dir.exists():
            return jsonify({
                "success": True,
                "configs": [],
                "message": "配置目录不存在"
            })
        
        # 获取所有yaml配置文件
        config_files = []
        for config_file in config_dir.glob('*.yaml'):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                config_info = {
                    "name": config_file.stem,
                    "path": str(config_file),
                    "model": config_data.get('MODEL', {}).get('NAME', 'Unknown') if config_data else 'Unknown',
                    "dataset": config_data.get('DATASET', {}).get('NAME', 'Unknown') if config_data else 'Unknown',
                    "modified": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat()
                }
                config_files.append(config_info)
            except Exception as e:
                logger.warning(f"读取配置文件 {config_file} 失败: {e}")
                continue
        
        return jsonify({
            "success": True,
            "configs": config_files,
            "message": f"找到 {len(config_files)} 个配置文件"
        })
        
    except Exception as e:
        logger.error(f"列出配置文件失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def run_validation(job: TrainingJob):
    """运行验证任务"""
    try:
        job.status = "running"
        job.start_time = datetime.now().isoformat()
        job.status_description = "正在初始化验证..."
        job.add_log(f"🚀 开始验证任务: {job.model_name}")
        
        # 在终端显示验证开始信息
        print(f"\n{'='*80}")
        print(f"🔬 vulcan 验证任务启动")
        print(f"{'='*80}")
        print(f"📋 任务详情:")
        print(f"   • 任务ID: {job.job_id}")
        print(f"   • 模型: {job.model_name}")
        print(f"   • 配置文件: {job.config_path}")
        print(f"   • 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 获取当前工作目录和项目根目录
        current_dir = os.path.dirname(__file__)
        project_root = current_dir
        
        job.add_log(f"📂 工作目录: {project_root}")
        print(f"📂 工作目录: {project_root}")
        
        # 构建验证命令 - 使用绝对路径
        val_script = os.path.join(project_root, "tools", "val.py")
        config_path = os.path.abspath(job.config_path)
        
        # 详细的文件存在性检查
        job.add_log(f"🔍 检查验证脚本: {val_script}")
        print(f"🔍 检查验证脚本: {val_script}")
        if not os.path.exists(val_script):
            error_msg = f"❌ 验证脚本不存在: {val_script}"
            job.add_log(error_msg)
            print(error_msg)
            job.status = "failed"
            job.status_description = "验证脚本不存在"
            return
            
        job.add_log(f"✅ 验证脚本存在")
        print(f"✅ 验证脚本存在")
        
        job.add_log(f"🔍 检查配置文件: {config_path}")
        print(f"🔍 检查配置文件: {config_path}")
        if not os.path.exists(config_path):
            error_msg = f"❌ 配置文件不存在: {config_path}"
            job.add_log(error_msg)
            print(error_msg)
            job.status = "failed"
            job.status_description = "配置文件不存在"
            return
            
        job.add_log(f"✅ 配置文件存在")
        print(f"✅ 配置文件存在")
        
        # 检查模型输出目录
        output_dir = os.path.join(project_root, "output")
        job.add_log(f"🔍 检查输出目录: {output_dir}")
        print(f"🔍 检查输出目录: {output_dir}")
        if os.path.exists(output_dir):
            model_files = list(Path(output_dir).glob("*.pth")) + list(Path(output_dir).glob("*.pt"))
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                job.add_log(f"✅ 找到模型文件: {latest_model}")
                print(f"✅ 找到模型文件: {latest_model}")
            else:
                job.add_log(f"⚠️ 输出目录中未找到模型文件 (.pth/.pt)")
                print(f"⚠️ 输出目录中未找到模型文件 (.pth/.pt)")
        else:
            job.add_log(f"⚠️ 输出目录不存在: {output_dir}")
            print(f"⚠️ 输出目录不存在: {output_dir}")
        
        # 构建验证命令
        cmd = [
            sys.executable,
            val_script,
            "--cfg", config_path
        ]
        
        job.add_log(f"🔧 执行验证命令: {' '.join(cmd)}")
        print(f"🔧 执行验证命令: {' '.join(cmd)}")
        job.status_description = "正在启动验证进程..."
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # 确保输出不被缓冲
        env['PYTHONIOENCODING'] = 'utf-8'  # 确保中文输出正常
        
        # 启动验证进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=project_root,
            env=dict(os.environ, PYTHONUNBUFFERED='1')
        )
        
        job.process = process
        
        print(f"✅ 验证进程已启动，PID: {process.pid}")
        job.add_log(f"✅ 验证进程已启动，PID: {process.pid}")
        print(f"\n📋 验证实时输出:")
        print(f"{'='*60}")
        
        # 监控验证输出
        try:
            for line in process.stdout:
                line = line.strip()
                if line:
                    # 同时在终端和日志中显示
                    print(f"🔬 {line}")
                    job.add_log(line)
                    
                    # 解析验证进度 - 处理类似 "39% | 886/2273" 的格式
                    if '%' in line and ('|' in line or '/' in line):
                        try:
                            # 解析百分比进度
                            percent_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                            if percent_match:
                                progress_percent = float(percent_match.group(1))
                                job.progress = progress_percent
                                print(f"📊 验证进度更新: {progress_percent:.1f}%")
                                job.add_log(f"验证进度更新: {progress_percent:.1f}%")
                            
                            # 解析步骤进度 (如: 886/2273)
                            step_match = re.search(r'(\d+)/(\d+)', line)
                            if step_match:
                                current_step = int(step_match.group(1))
                                total_steps = int(step_match.group(2))
                                job.current_iteration = current_step
                                job.total_iterations = total_steps
                                if total_steps > 0:
                                    step_progress = (current_step / total_steps) * 100
                                    # 如果没有找到百分比，使用步骤计算的进度
                                    if job.progress == 0:
                                        job.progress = step_progress
                                        print(f"📈 验证步骤进度: {current_step}/{total_steps} ({step_progress:.1f}%)")
                                        job.add_log(f"验证步骤进度: {current_step}/{total_steps} ({step_progress:.1f}%)")
                                    else:
                                        print(f"📈 验证步骤: {current_step}/{total_steps}")
                                        job.add_log(f"验证步骤: {current_step}/{total_steps}")
                        except Exception as e:
                            job.add_log(f"解析验证进度失败: {e}")
                    
                    # 解析验证指标 - 更加灵活的解析
                    if "Accuracy:" in line or "accuracy:" in line or "acc:" in line:
                        try:
                            # 支持多种格式: Accuracy: 0.85, accuracy: 85%, acc: 0.8500
                            patterns = [r'Accuracy[:\s]+([\d\.]+)', r'accuracy[:\s]+([\d\.]+)', r'acc[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                acc_match = re.search(pattern, line, re.IGNORECASE)
                                if acc_match:
                                    acc_value = float(acc_match.group(1))
                                    # 如果值大于1，可能是百分比格式
                                    if acc_value > 1:
                                        acc_value = acc_value / 100
                                    job.metrics["validation_accuracy"] = acc_value
                                    print(f"🎯 解析到验证准确率: {acc_value:.4f}")
                                    job.add_log(f"解析到验证准确率: {acc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析accuracy信息失败: {e} - 原始行: {line}")
                    
                    if "F1:" in line or "f1:" in line:
                        try:
                            patterns = [r'F1[:\s]+([\d\.]+)', r'f1[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                f1_match = re.search(pattern, line, re.IGNORECASE)
                                if f1_match:
                                    f1_value = float(f1_match.group(1))
                                    if f1_value > 1:
                                        f1_value = f1_value / 100
                                    job.metrics["validation_f1"] = f1_value
                                    print(f"📊 解析到验证F1: {f1_value:.4f}")
                                    job.add_log(f"解析到验证F1: {f1_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析F1信息失败: {e} - 原始行: {line}")
                    
                    if "Precision:" in line or "precision:" in line or "prec:" in line:
                        try:
                            patterns = [r'Precision[:\s]+([\d\.]+)', r'precision[:\s]+([\d\.]+)', r'prec[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                prec_match = re.search(pattern, line, re.IGNORECASE)
                                if prec_match:
                                    prec_value = float(prec_match.group(1))
                                    if prec_value > 1:
                                        prec_value = prec_value / 100
                                    job.metrics["validation_precision"] = prec_value
                                    print(f"📈 解析到验证精确率: {prec_value:.4f}")
                                    job.add_log(f"解析到验证精确率: {prec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析precision信息失败: {e} - 原始行: {line}")
                    
                    if "Recall:" in line or "recall:" in line or "rec:" in line:
                        try:
                            patterns = [r'Recall[:\s]+([\d\.]+)', r'recall[:\s]+([\d\.]+)', r'rec[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                rec_match = re.search(pattern, line, re.IGNORECASE)
                                if rec_match:
                                    rec_value = float(rec_match.group(1))
                                    if rec_value > 1:
                                        rec_value = rec_value / 100
                                    job.metrics["validation_recall"] = rec_value
                                    print(f"🔄 解析到验证召回率: {rec_value:.4f}")
                                    job.add_log(f"解析到验证召回率: {rec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析recall信息失败: {e} - 原始行: {line}")
                    
                    if "ROC_AUC:" in line or "AUC:" in line or "auc:" in line:
                        try:
                            patterns = [r'ROC_AUC[:\s]+([\d\.]+)', r'AUC[:\s]+([\d\.]+)', r'auc[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                auc_match = re.search(pattern, line, re.IGNORECASE)
                                if auc_match:
                                    auc_value = float(auc_match.group(1))
                                    if auc_value > 1:
                                        auc_value = auc_value / 100
                                    job.metrics["validation_auc"] = auc_value
                                    print(f"📌 解析到验证AUC: {auc_value:.4f}")
                                    job.add_log(f"解析到验证AUC: {auc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析AUC信息失败: {e} - 原始行: {line}")
                    
                    # 检测错误信息
                    if any(error_word in line.lower() for error_word in ['error', 'exception', 'failed', 'traceback']):
                        print(f"❌ 检测到错误信息: {line}")
                        job.add_log(f"检测到错误信息: {line}")
        
        except Exception as e:
            error_msg = f"读取验证输出时发生错误: {e}"
            job.add_log(error_msg)
            print(f"❌ {error_msg}")
        
        print(f"{'='*60}")
        print(f"📊 等待验证进程完成...")
        
        # 等待进程结束
        return_code = process.wait()
        
        print(f"📊 验证进程结束，返回码: {return_code}")
        
        # 判断验证是否成功 - 考虑返回码和是否有指标
        has_metrics = any(key.startswith('validation_') for key in job.metrics.keys())
        
        if return_code == 0 or has_metrics:
            job.status = "completed"
            job.add_log("验证完成！")
            print(f"✅ 验证完成！")
            
            # 确保进度为100%
            job.progress = 100
            
            # 显示验证结果摘要
            print(f"\n🎯 验证结果摘要:")
            job.add_log("验证结果摘要:")
            if job.metrics.get("validation_accuracy", 0) > 0:
                acc_msg = f"• 验证准确率: {job.metrics['validation_accuracy']:.4f}"
                job.add_log(acc_msg)
                print(f"  {acc_msg}")
            if job.metrics.get("validation_f1", 0) > 0:
                f1_msg = f"• 验证F1: {job.metrics['validation_f1']:.4f}"
                job.add_log(f1_msg)
                print(f"  {f1_msg}")
            if job.metrics.get("validation_precision", 0) > 0:
                prec_msg = f"• 验证精确率: {job.metrics['validation_precision']:.4f}"
                job.add_log(prec_msg)
                print(f"  {prec_msg}")
            if job.metrics.get("validation_recall", 0) > 0:
                rec_msg = f"• 验证召回率: {job.metrics['validation_recall']:.4f}"
                job.add_log(rec_msg)
                print(f"  {rec_msg}")
            if job.metrics.get("validation_auc", 0) > 0:
                auc_msg = f"• 验证AUC: {job.metrics['validation_auc']:.4f}"
                job.add_log(auc_msg)
                print(f"  {auc_msg}")
                
            # 如果没有解析到指标，给出提示
            if not has_metrics:
                warning_msg = "注意: 未解析到验证指标，请检查验证脚本输出格式"
                job.add_log(warning_msg)
                print(f"⚠️ {warning_msg}")
        else:
            job.status = "failed"
            failure_msg = f"验证失败，返回码: {return_code}"
            job.add_log(failure_msg)
            print(f"❌ {failure_msg}")
            # 添加更详细的错误信息
            job.add_log("验证进程异常退出，请检查:")
            job.add_log("1. 模型文件是否存在")
            job.add_log("2. 配置文件路径是否正确")
            job.add_log("3. 验证数据集是否可访问")
            job.add_log("4. 系统内存是否充足")
            print(f"📋 故障排除建议:")
            print(f"  1. 模型文件是否存在")
            print(f"  2. 配置文件路径是否正确")
            print(f"  3. 验证数据集是否可访问")
            print(f"  4. 系统内存是否充足")
        
        job.end_time = datetime.now().isoformat()
        duration_msg = f"⏱️ 验证用时: {job.get_duration()}"
        job.add_log(duration_msg)
        print(f"{duration_msg}")
        
        print(f"{'='*80}")
        print(f"🔬 vulcan 验证任务完成")
        print(f"{'='*80}\n")
        
        # 确保日志保存到文件
        job.save_logs_to_file()
        
    except Exception as e:
        job.status = "failed"
        job.status_description = f"验证异常: {str(e)}"
        job.add_log(f"❌ 验证异常: {str(e)}")
        job.end_time = datetime.now().isoformat()
        logger.error(f"验证任务 {job.job_id} 异常: {e}", exc_info=True)
        print(f"💥 [验证异常] {str(e)}")
        
        # 添加详细的异常信息到日志
        import traceback
        job.add_log("📝 详细异常信息:")
        job.add_log(traceback.format_exc())
        
        print(f"📝 详细异常信息:")
        print(traceback.format_exc())
        
        # 确保日志保存到文件
        job.save_logs_to_file()

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "success": True,
        "message": "vulcan Backend API Server is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/start-validation/<config_name>', methods=['POST'])
def start_validation_with_config(config_name: str):
    """使用现有配置文件启动验证"""
    try:
        print(f"\n🔬 [API] 接收到验证启动请求")
        print(f"📋 配置文件: {config_name}.yaml")
        
        # 检查验证脚本格式
        if not ensure_validation_script_format():
            logger.warning("验证脚本格式检查失败，但继续执行验证")
            print(f"⚠️ 验证脚本格式检查失败，但继续执行验证")
        
        # 构建配置文件路径
        config_path = os.path.join('configs', f'{config_name}.yaml')
        
        if not os.path.exists(config_path):
            error_msg = f"配置文件 {config_name}.yaml 不存在"
            print(f"❌ {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        # 读取配置文件获取模型名称
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            model_name = config_data.get('MODEL', {}).get('NAME', config_name)
        except:
            model_name = config_name
        
        print(f"🤖 验证模型: {model_name}")
        
        # 生成验证任务ID
        job_id = str(uuid.uuid4())
        print(f"🆔 验证任务ID: {job_id}")
        
        # 创建验证任务
        job = TrainingJob(
            job_id=job_id,
            config_path=config_path,
            model_name=model_name
        )
        
        # 设置验证任务的特殊标识
        job.is_validation_task = True
        job.total_epochs = 1  # 验证任务通常只需要1个epoch
        job.progress = 0  # 设置初始进度为0
        
        training_jobs[job_id] = job
        
        print(f"🚀 启动验证线程...")
        
        # 启动验证线程
        validation_thread = threading.Thread(
            target=run_validation,
            args=(job,),
            daemon=True
        )
        validation_thread.start()
        
        success_msg = f"验证任务已启动: {model_name} (使用 {config_name}.yaml)"
        print(f"✅ {success_msg}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "config_name": config_name,
            "config_path": config_path,
            "model_name": model_name,
            "task_type": "validation",
            "message": success_msg
        })
        
    except Exception as e:
        error_msg = f"启动验证失败: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/start-validation-with-config-id/<config_id>', methods=['POST'])
def start_validation_with_config_id(config_id: str):
    """使用配置ID启动验证（适用于generated_configs中的配置）"""
    try:
        print(f"\n🔬 [API] 接收到验证启动请求（配置ID）")
        print(f"🆔 配置ID: {config_id}")
        
        # 检查验证脚本格式
        if not ensure_validation_script_format():
            logger.warning("验证脚本格式检查失败，但继续执行验证")
            print(f"⚠️ 验证脚本格式检查失败，但继续执行验证")
        
        # 检查配置ID是否存在
        if config_id not in training_configs:
            error_msg = f"配置ID {config_id} 不存在"
            print(f"❌ {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        config_info = training_configs[config_id]
        config_path = config_info['config_path']
        model_name = config_info['model_name']
        
        print(f"📄 配置文件路径: {config_path}")
        print(f"🤖 验证模型: {model_name}")
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            error_msg = f"配置文件 {config_path} 不存在"
            print(f"❌ {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        # 生成验证任务ID
        job_id = str(uuid.uuid4())
        print(f"🆔 验证任务ID: {job_id}")
        
        # 创建验证任务
        job = TrainingJob(
            job_id=job_id,
            config_path=config_path,
            model_name=model_name
        )
        
        # 设置验证任务的特殊标识
        job.is_validation_task = True
        job.total_epochs = 1  # 验证任务通常只需要1个epoch
        job.progress = 0  # 设置初始进度为0
        
        training_jobs[job_id] = job
        
        print(f"🚀 启动验证线程...")
        
        # 启动验证线程
        validation_thread = threading.Thread(
            target=run_validation,
            args=(job,),
            daemon=True
        )
        validation_thread.start()
        
        success_msg = f"验证任务已启动: {model_name} (使用配置ID {config_id})"
        print(f"✅ {success_msg}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "config_id": config_id,
            "config_path": config_path,
            "model_name": model_name,
            "filename": config_info.get('filename', 'unknown'),
            "task_type": "validation",
            "message": success_msg
        })
        
    except Exception as e:
        error_msg = f"使用配置ID启动验证失败: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/validation-status/<job_id>', methods=['GET'])
def get_validation_status(job_id: str):
    """获取验证状态"""
    try:
        print(f"🔍 [API] 查询验证状态: {job_id}")
        
        if job_id not in training_jobs:
            error_msg = "验证任务不存在"
            print(f"❌ {error_msg}: {job_id}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        job = training_jobs[job_id]
        
        # 检查是否是验证任务 - 改进检查逻辑
        is_validation_task = (
            hasattr(job, 'is_validation_task') and job.is_validation_task
        ) or (
            # 如果没有明确标记但有验证指标，也认为是验证任务
            any(key.startswith('validation_') for key in job.metrics.keys())
        )
        
        print(f"📊 验证状态: {job.status}, 进度: {job.progress}%, 是否验证任务: {is_validation_task}")
        
        # 分离验证指标
        validation_metrics = {}
        for key, value in job.metrics.items():
            if key.startswith('validation_'):
                validation_metrics[key] = value
        
        if validation_metrics:
            print(f"🎯 当前验证指标: {validation_metrics}")
        
        # 获取日志
        logs = job.get_recent_logs(100)
        
        # 构建响应数据
        response_data = {
            "success": True,
            "job_id": job_id,
            "task_type": "validation",
            "status": job.status,
            "progress": job.progress,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "current_iteration": job.current_iteration,  # 添加当前迭代
            "total_iterations": job.total_iterations,    # 添加总迭代数
            "start_time": job.start_time,
            "end_time": job.end_time,
            "logs": logs,
            "log_count": len(logs),
            "log_file_path": job.get_log_file_path(),
            
            # 验证指标
            "validation_metrics": validation_metrics,
            
            # 任务信息
            "model_name": job.model_name,
            "config_path": job.config_path,
            
            # 状态描述
            "status_description": get_validation_status_description(job.status),
            
            # 验证结果摘要
            "validation_summary": generate_validation_summary(validation_metrics)
        }
        
        # 如果验证已完成，添加最终结果
        if job.status == "completed":
            response_data["final_validation_results"] = generate_final_validation_results(validation_metrics, job)
            print(f"✅ 验证已完成，返回最终结果")
        
        # 如果任务失败，添加错误详情和故障排除建议
        if job.status == "failed":
            error_logs = [log for log in logs if any(keyword in log.lower() for keyword in ['error', 'exception', 'failed', 'traceback', '错误', '失败', '异常'])]
            response_data["error_details"] = {
                "error_logs": error_logs[-10:] if error_logs else [],  # 最后10条错误日志
                "troubleshooting_tips": [
                    "检查模型文件是否存在于 output 目录",
                    "确认配置文件中的路径设置正确",
                    "验证数据集路径是否可访问",
                    "检查系统内存是否充足",
                    "查看完整日志获取详细错误信息"
                ]
            }
            print(f"❌ 验证失败，错误日志数量: {len(error_logs)}")
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"获取验证状态失败: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def get_validation_status_description(status: str) -> str:
    """获取验证状态描述"""
    if status == "pending":
        return "验证等待中"
    elif status == "running":
        return "验证进行中"
    elif status == "completed":
        return "验证已完成"
    elif status == "failed":
        return "验证失败"
    else:
        return "未知状态"

def generate_validation_summary(validation_metrics: dict) -> dict:
    """生成验证摘要"""
    summary = {
        "accuracy": None,
        "f1": None,
        "precision": None,
        "recall": None,
        "auc": None,
        "overall_score": None
    }
    
    if validation_metrics:
        if "validation_accuracy" in validation_metrics:
            summary["accuracy"] = {
                "value": validation_metrics["validation_accuracy"],
                "percentage": validation_metrics["validation_accuracy"] * 100,
                "formatted": f"{validation_metrics['validation_accuracy']:.4f} ({validation_metrics['validation_accuracy']*100:.1f}%)"
            }
        
        if "validation_f1" in validation_metrics:
            summary["f1"] = {
                "value": validation_metrics["validation_f1"],
                "percentage": validation_metrics["validation_f1"] * 100,
                "formatted": f"{validation_metrics['validation_f1']:.4f} ({validation_metrics['validation_f1']*100:.1f}%)"
            }
        
        if "validation_precision" in validation_metrics:
            summary["precision"] = {
                "value": validation_metrics["validation_precision"],
                "percentage": validation_metrics["validation_precision"] * 100,
                "formatted": f"{validation_metrics['validation_precision']:.4f} ({validation_metrics['validation_precision']*100:.1f}%)"
            }
        
        if "validation_recall" in validation_metrics:
            summary["recall"] = {
                "value": validation_metrics["validation_recall"],
                "percentage": validation_metrics["validation_recall"] * 100,
                "formatted": f"{validation_metrics['validation_recall']:.4f} ({validation_metrics['validation_recall']*100:.1f}%)"
            }
        
        if "validation_auc" in validation_metrics:
            summary["auc"] = {
                "value": validation_metrics["validation_auc"],
                "formatted": f"{validation_metrics['validation_auc']:.4f}"
            }
        
        if "validation_overall_score" in validation_metrics:
            summary["overall_score"] = {
                "value": validation_metrics["validation_overall_score"],
                "percentage": validation_metrics["validation_overall_score"] * 100,
                "formatted": f"{validation_metrics['validation_overall_score']:.4f} ({validation_metrics['validation_overall_score']*100:.1f}%)"
            }
    
    return summary

def generate_final_validation_results(validation_metrics: dict, job) -> dict:
    """生成最终验证结果"""
    final_results = {
        "completion_time": job.end_time,
        "duration": job.get_duration(),
        "model_name": job.model_name,
        "config_path": job.config_path,
        "validation_completed": True,
        "key_metrics": {},
        "performance_assessment": [],
        "recommendations": []
    }
    
    # 关键指标 - 返回原始数值格式（用户要求的格式）
    if validation_metrics:
        if "validation_accuracy" in validation_metrics:
            final_results["key_metrics"]["accuracy"] = validation_metrics["validation_accuracy"]
            final_results["key_metrics"]["best_accuracy"] = validation_metrics["validation_accuracy"]
        
        if "validation_f1" in validation_metrics:
            final_results["key_metrics"]["f1"] = validation_metrics["validation_f1"]
        
        if "validation_loss" in validation_metrics:
            final_results["key_metrics"]["loss"] = validation_metrics["validation_loss"]
        
        if "validation_precision" in validation_metrics:
            final_results["key_metrics"]["precision"] = validation_metrics["validation_precision"]
        
        if "validation_recall" in validation_metrics:
            final_results["key_metrics"]["recall"] = validation_metrics["validation_recall"]
        
        if "validation_auc" in validation_metrics:
            final_results["key_metrics"]["auc"] = validation_metrics["validation_auc"]
        
        if "validation_overall_score" in validation_metrics:
            final_results["key_metrics"]["overall_score"] = validation_metrics["validation_overall_score"]
    
    # 性能评估
    if validation_metrics:
        acc = validation_metrics.get("validation_accuracy", 0)
        f1 = validation_metrics.get("validation_f1", 0)
        
        if acc > 0.9:
            final_results["performance_assessment"].append("🎉 模型性能优秀！准确率超过90%")
        elif acc > 0.8:
            final_results["performance_assessment"].append("👍 模型性能良好，准确率超过80%")
        elif acc > 0.7:
            final_results["performance_assessment"].append("⚠️ 模型性能一般，准确率在70-80%之间")
        else:
            final_results["performance_assessment"].append("❌ 模型性能较差，准确率低于70%")
        
        if f1 > 0 and abs(acc - f1) > 0.1:
            final_results["performance_assessment"].append("📊 准确率和F1分数差异较大，可能存在数据不平衡问题")
    
    # 建议
    if validation_metrics:
        acc = validation_metrics.get("validation_accuracy", 0)
        f1 = validation_metrics.get("validation_f1", 0)
        
        if acc > 0.9:
            final_results["recommendations"].append("🚀 模型性能优秀，可以直接部署使用")
        elif acc > 0.8:
            final_results["recommendations"].append("✅ 模型性能良好，建议在生产环境中测试")
        elif acc > 0.7:
            final_results["recommendations"].append("🔧 建议调整超参数或增加训练数据来提升性能")
        else:
            final_results["recommendations"].append("⚠️ 建议重新设计模型架构或检查数据质量")
        
        if f1 > 0 and f1 < 0.6:
            final_results["recommendations"].append("📈 建议关注数据平衡性，可能需要调整损失函数")
    
    return final_results

class OptimizationJob:
    """数据集优化任务类"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"  # pending, running, completed, failed
        self.status_description = "等待开始优化..."
        self.progress = 0
        self.current_iteration = 0
        self.total_iterations = 15
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.process = None
        self.best_ratio = 0.0
        self.best_f1_score = 0.0
        self.current_ratio = 0.0  # 当前测试的比例
        self.current_f1 = 0.0    # 当前测试的F1分数
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
            logger.error(f"写入优化日志文件失败: {e}")
        
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
            logger.error(f"读取优化日志文件失败: {e}")
        
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
        job.status_description = "正在优化数据集..."
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
            cwd=project_root,
            env=dict(os.environ, PYTHONUNBUFFERED='1')  # 确保Python输出不被缓冲
        )
        
        job.process = process
        
        print(f"✅ 子进程已启动，PID: {process.pid}")
        job.add_log(f"✅ 子进程已启动，PID: {process.pid}")
        
        # 实时读取输出并解析
        patterns = {
            'iteration': r'第(\d+)次迭代',
            'ratio': r'比例\s+([0-9.]+)',
            'f1_score': r'F1分数:\s+([0-9.]+)',
            'best_ratio': r'最佳比例:\s+([0-9.]+)',
            'best_f1': r'F1分数:\s+([0-9.]+)',
            'progress': r'进度[：:]\s*(\d+)%',
            'completed': r'搜索完成',
            'failed': r'失败|错误|Error|Exception'
        }
        
        current_ratio = 0.0
        last_output_time = time.time()
        no_output_timeout = 60  # 60秒无输出超时
        
        # 实时读取输出
        while True:
            # 检查进程是否还在运行
            if process.poll() is not None:
                print(f"📊 进程已结束，返回码: {process.returncode}")
                job.add_log(f"📊 进程已结束，返回码: {process.returncode}")
                break
            
            # 检查超时
            if time.time() - last_output_time > no_output_timeout:
                print(f"⚠️ 警告: {no_output_timeout}秒无输出，可能卡住了")
                job.add_log(f"⚠️ 警告: {no_output_timeout}秒无输出，可能卡住了")
                # 不终止进程，继续等待
            
            # 尝试读取输出
            try:
                line = process.stdout.readline()
                if line:
                    last_output_time = time.time()
                    # 同时输出到控制台和日志
                    print(line.rstrip())
                    job.add_log(line.rstrip())
                    
                    # 解析进度信息
                    for key, pattern in patterns.items():
                        match = re.search(pattern, line)
                        if match:
                            if key == 'iteration':
                                job.current_iteration = int(match.group(1))
                                job.progress = min(100, int((job.current_iteration / job.total_iterations) * 100))
                                print(f"📊 更新进度: {job.progress}% (第{job.current_iteration}次迭代)")
                            elif key == 'ratio':
                                job.current_ratio = float(match.group(1))
                                print(f"📈 当前测试比例: {job.current_ratio:.3f}")
                            elif key == 'f1_score':
                                job.current_f1 = float(match.group(1))
                                print(f"📈 当前F1分数: {job.current_f1:.4f}")
                                if job.current_f1 > job.best_f1_score:
                                    job.best_f1_score = job.current_f1
                                    job.best_ratio = job.current_ratio
                                    print(f"🏆 新的最佳记录: 比例={job.best_ratio:.3f}, F1={job.best_f1_score:.4f}")
                            elif key == 'best_ratio':
                                job.best_ratio = float(match.group(1))
                            elif key == 'completed':
                                job.status = "completed"
                                job.status_description = "数据集优化完成"
                                job.progress = 100
                                print("✅ 优化任务完成")
                            elif key == 'failed':
                                job.status = "failed"
                                job.status_description = "数据集优化失败"
                                print("❌ 检测到错误")
                else:
                    # 没有输出，短暂等待
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ 读取输出时出错: {e}")
                job.add_log(f"❌ 读取输出时出错: {e}")
                break
        
        # 等待进程完成
        try:
            return_code = process.wait(timeout=30)  # 30秒超时
        except subprocess.TimeoutExpired:
            print("⚠️ 进程等待超时，强制终止")
            job.add_log("⚠️ 进程等待超时，强制终止")
            process.kill()
            return_code = -1
        
        job.end_time = datetime.now().isoformat()
        
        print(f"📊 子进程结束，返回码: {return_code}")
        job.add_log(f"📊 子进程结束，返回码: {return_code}")
        
        if return_code == 0 and job.status != "failed":
            job.status = "completed"
            job.status_description = f"优化完成，最佳比例: {job.best_ratio:.3f}, F1: {job.best_f1_score:.4f}"
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
            job.status_description = f"优化失败，返回码: {return_code}"
            error_msg = f"\n❌ 数据集优化任务失败！返回码: {return_code}"
            print(error_msg)
            job.add_log(error_msg)
            
    except Exception as e:
        job.status = "failed"
        job.status_description = f"优化异常: {str(e)}"
        job.end_time = datetime.now().isoformat()
        error_msg = f"数据集优化任务异常: {str(e)}"
        print(error_msg)
        job.add_log(error_msg)
        logger.error(f"数据集优化任务异常: {e}")
        import traceback
        job.add_log(traceback.format_exc())

@app.route('/api/start-dataset-optimization', methods=['POST'])
def start_dataset_optimization():
    """启动数据集优化"""
    try:
        data = request.json
        max_iterations = data.get('max_iterations', 15)
        
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
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "数据集优化任务已启动"
        })
        
    except Exception as e:
        logger.error(f"启动数据集优化失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/optimization-status/<job_id>', methods=['GET'])
def get_optimization_status(job_id: str):
    """获取数据集优化状态"""
    try:
        if job_id not in optimization_jobs:
            return jsonify({
                "success": False,
                "error": "优化任务不存在"
            }), 404
        
        job = optimization_jobs[job_id]
        
        # 根据任务状态决定返回的日志数量
        if job.status == "completed":
            logs = job.get_full_logs()
            log_count = len(logs)
        elif job.status == "failed":
            logs = job.get_full_logs()
            log_count = len(logs)
        else:
            # 运行中时返回更多日志以便实时显示
            logs = job.get_logs(limit=100)  # 返回最近100条日志
            log_count = len(logs)
        
        # 构建响应数据
        response_data = {
            "success": True,
            "job_id": job_id,
            "status": job.status,
            "status_description": job.status_description,
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
            "best_f1_score": job.best_f1_score,
            "current_ratio": job.current_ratio,
            "current_f1": job.current_f1
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"获取优化状态失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/optimization-logs/<job_id>', methods=['GET'])
def get_optimization_logs(job_id: str):
    """获取数据集优化日志"""
    try:
        if job_id not in optimization_jobs:
            return jsonify({
                "success": False,
                "error": "优化任务不存在"
            }), 404
        
        job = optimization_jobs[job_id]
        logs = job.get_full_logs()  # 获取完整日志
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "logs": logs,
            "log_count": len(logs),
            "status": job.status,
            "duration": job.get_duration(),
            "best_ratio": job.best_ratio,
            "best_f1_score": job.best_f1_score
        })
        
    except Exception as e:
        logger.error(f"获取优化日志失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

class PaperSearchJob:
    def __init__(self, job_id, search_query="deep learning vulnerability detection", year_from=2023, top_k=5):
        self.job_id = job_id
        self.status = 'running'
        self.search_query = search_query
        self.year_from = year_from
        self.top_k = top_k
        self.papers = []
        self.logs = []
        self.status_description = '正在检索论文...'
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = 0
        self.progress = 0  # 添加进度字段

    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

def run_paper_search(job: PaperSearchJob):
    try:
        job.status = 'running'
        job.status_description = '正在检索论文...'
        job.start_time = datetime.now()
        job.add_log("开始检索论文...")
        job.progress = 0
        
        # 导入ResourceUpdater类
        from resource_updater import ResourceUpdater
        
        # 创建ResourceUpdater实例
        updater = ResourceUpdater()
        
        job.add_log(f"使用查询词: {job.search_query}")
        job.add_log(f"起始年份: {job.year_from}")
        job.add_log(f"返回数量: {job.top_k}")
        job.progress = 20
        
        # 调用论文检索功能
        job.add_log("正在连接学术数据库...")
        job.progress = 40
        
        papers = updater.semantic_search_papers(
            query=job.search_query,
            year_from=job.year_from,
            top_k=job.top_k
        )
        
        job.progress = 80
        job.add_log(f"检索完成，找到 {len(papers)} 篇论文")
        
        # 处理检索结果
        for i, paper in enumerate(papers, 1):
            job.add_log(f"论文 {i}: {paper.get('title', 'Unknown Title')}")
            job.papers.append(paper)
        
        job.progress = 100
        job.status = 'completed'
        job.status_description = f'检索完成，找到 {len(papers)} 篇论文'
        job.end_time = datetime.now()
        job.duration = (job.end_time - job.start_time).total_seconds()
        job.add_log("论文检索完成！")
        
    except Exception as e:
        job.status = 'failed'
        job.status_description = f'检索失败: {str(e)}'
        job.end_time = datetime.now()
        job.duration = (job.end_time - job.start_time).total_seconds()
        job.progress = 0
        job.add_log(f"论文检索失败: {e}")
        import traceback
        job.add_log(f"错误详情: {traceback.format_exc()}")

@app.route('/api/start-paper-search', methods=['POST'])
def start_paper_search():
    try:
        data = request.json
        search_query = data.get('search_query', 'deep learning vulnerability detection')
        year_from = data.get('year_from', 2023)
        top_k = data.get('top_k', 5)
        
        job_id = str(uuid.uuid4())
        job = PaperSearchJob(job_id, search_query, year_from, top_k)
        
        # 将job添加到全局jobs字典
        jobs[job_id] = job
        
        # 启动论文检索线程
        search_thread = threading.Thread(target=run_paper_search, args=(job,))
        search_thread.start()
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": job.status,
            "status_description": job.status_description,
            "start_time": job.start_time.isoformat(),
            "duration": job.duration
        })
    except Exception as e:
        logger.error(f"启动论文检索失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/paper-search-status/<job_id>', methods=['GET'])
def get_paper_search_status(job_id: str):
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "论文检索任务不存在"
            }), 404
        
        job = jobs[job_id]
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": job.status,
            "status_description": job.status_description,
            "progress": job.progress,
            "start_time": job.start_time.isoformat(),
            "duration": job.duration,
            "search_query": job.search_query,
            "year_from": job.year_from,
            "top_k": job.top_k,
            "papers": job.papers,
            "logs": job.logs,
            "log_count": len(job.logs)
        })
    except Exception as e:
        logger.error(f"获取论文检索状态失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/paper-search-logs/<job_id>', methods=['GET'])
def get_paper_search_logs(job_id: str):
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "论文检索任务不存在"
            }), 404
        
        job = jobs[job_id]
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "logs": job.logs
        })
    except Exception as e:
        logger.error(f"获取论文检索日志失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/papers/<job_id>', methods=['GET'])
def get_papers(job_id: str):
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "论文检索任务不存在"
            }), 404
        
        job = jobs[job_id]
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "papers": job.papers
        })
    except Exception as e:
        logger.error(f"获取论文失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# 在文件末尾添加数据收集相关的API端点

class DataCollectionJob:
    """数据收集任务类"""
    def __init__(self, job_id: str, collection_type: str = "comprehensive"):
        self.job_id = job_id
        self.collection_type = collection_type
        self.status = "pending"  # pending, running, completed, failed
        self.status_description = "等待启动"
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.collection_results = {}
        self.error = None
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """确保日志目录存在"""
        import os
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def add_log(self, message: str):
        """添加日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # 限制日志数量
        if len(self.logs) > 1000:
            self.logs = self.logs[-500:]
    
    def get_logs(self, limit: int = None) -> List[str]:
        """获取日志"""
        if limit:
            return self.logs[-limit:]
        return self.logs
    
    def get_recent_logs(self, count: int = 50) -> List[str]:
        """获取最近的日志"""
        return self.logs[-count:]
    
    def get_duration(self) -> str:
        """获取任务持续时间"""
        if not self.start_time:
            return "未开始"
        if not self.end_time:
            end_time = datetime.now()
        else:
            end_time = self.end_time
        
        duration = end_time - self.start_time
        return str(duration).split('.')[0]  # 移除微秒部分

def run_data_collection(job: DataCollectionJob):
    """运行数据收集任务"""
    try:
        job.status = "running"
        job.status_description = "正在收集数据"
        job.start_time = datetime.now()
        job.add_log("🚀 开始数据收集任务")
        
        # 导入数据收集器
        from data_collector import DataCollector
        
        job.add_log("📦 初始化数据收集器...")
        collector = DataCollector(job.collection_type)
        
        job.add_log("🔍 开始执行数据收集...")
        
        # 执行数据收集
        result = collector.collect_sample_data()
        
        if result["success"]:
            job.status = "completed"
            job.status_description = "数据收集完成"
            job.collection_results = result
            job.add_log("✅ 数据收集任务完成")
            job.add_log(f"📊 收集统计: {result['stats']}")
            job.add_log(f"📁 输出文件: {result['output_file']}")
        else:
            job.status = "failed"
            job.status_description = "数据收集失败"
            job.error = "数据收集执行失败"
            job.add_log("❌ 数据收集任务失败")
            
    except Exception as e:
        job.status = "failed"
        job.status_description = f"数据收集出错: {str(e)}"
        job.error = str(e)
        job.add_log(f"❌ 数据收集异常: {str(e)}")
        logger.error(f"数据收集任务异常: {e}")
        traceback.print_exc()
    
    finally:
        job.end_time = datetime.now()
        if job.start_time:
            job.duration = (job.end_time - job.start_time).total_seconds()

@app.route('/api/start-data-collection', methods=['POST'])
def start_data_collection():
    """启动数据收集任务"""
    try:
        data = request.get_json()
        collection_type = data.get('collection_type', 'comprehensive')
        
        # 生成任务ID
        job_id = str(uuid.uuid4())
        
        # 创建数据收集任务
        job = DataCollectionJob(job_id, collection_type)
        jobs[job_id] = job
        
        # 在后台线程中运行数据收集
        thread = threading.Thread(target=run_data_collection, args=(job,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"数据收集任务已启动: {job_id}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": f"数据收集任务已启动，任务ID: {job_id}",
            "collection_type": collection_type
        })
        
    except Exception as e:
        logger.error(f"启动数据收集失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-collection-status/<job_id>', methods=['GET'])
def get_data_collection_status(job_id: str):
    """获取数据收集状态"""
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "数据收集任务不存在"
            }), 404
        
        job = jobs[job_id]
        
        # 获取最近的日志
        recent_logs = job.get_recent_logs(20)
        
        response_data = {
            "success": True,
            "job_id": job_id,
            "status": job.status,
            "status_description": job.status_description,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "duration": job.get_duration(),
            "log_count": len(job.logs),
            "recent_logs": recent_logs,
            "collection_type": job.collection_type
        }
        
        # 如果任务完成，添加结果信息
        if job.status == "completed" and job.collection_results:
            response_data["collection_results"] = job.collection_results
        
        # 如果任务失败，添加错误信息
        if job.status == "failed" and job.error:
            response_data["error"] = job.error
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"获取数据收集状态失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-collection-logs/<job_id>', methods=['GET'])
def get_data_collection_logs(job_id: str):
    """获取数据收集日志"""
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "数据收集任务不存在"
            }), 404
        
        job = jobs[job_id]
        
        # 获取查询参数
        limit = request.args.get('limit', type=int)
        
        logs = job.get_logs(limit)
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "logs": logs,
            "total_logs": len(job.logs)
        })
        
    except Exception as e:
        logger.error(f"获取数据收集日志失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-collection-results/<job_id>', methods=['GET'])
def get_data_collection_results(job_id: str):
    """获取数据收集结果"""
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "数据收集任务不存在"
            }), 404
        
        job = jobs[job_id]
        
        if job.status != "completed":
            return jsonify({
                "success": False,
                "error": "数据收集任务尚未完成"
            }), 400
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "results": job.collection_results
        })
        
    except Exception as e:
        logger.error(f"获取数据收集结果失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def ensure_validation_script_format():
    """确保验证脚本输出格式正确"""
    try:
        val_script_path = os.path.join(os.path.dirname(__file__), "tools", "val.py")
        if not os.path.exists(val_script_path):
            logger.warning(f"验证脚本不存在: {val_script_path}")
            return False
        
        # 读取验证脚本内容
        with open(val_script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含正确的指标输出格式
        required_patterns = [
            'Accuracy:',
            'F1:',
            'Precision:',
            'Recall:'
        ]
        
        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            logger.warning(f"验证脚本缺少以下指标输出: {missing_patterns}")
            return False
        
        logger.info("验证脚本格式检查通过")
        return True
        
    except Exception as e:
        logger.error(f"检查验证脚本格式失败: {e}")
        return False

@app.route('/api/test-validation', methods=['GET'])
def test_validation_setup():
    """测试验证环境和配置"""
    try:
        test_results = {
            "validation_script_exists": False,
            "validation_script_format_ok": False,
            "output_directory_exists": False,
            "model_files_found": [],
            "config_files_found": [],
            "python_executable": sys.executable,
            "working_directory": os.path.dirname(__file__),
            "issues": [],
            "recommendations": []
        }
        
        # 检查验证脚本
        val_script_path = os.path.join(os.path.dirname(__file__), "tools", "val.py")
        test_results["validation_script_exists"] = os.path.exists(val_script_path)
        if not test_results["validation_script_exists"]:
            test_results["issues"].append("验证脚本 tools/val.py 不存在")
            test_results["recommendations"].append("确保 tools/val.py 文件存在")
        else:
            test_results["validation_script_format_ok"] = ensure_validation_script_format()
            if not test_results["validation_script_format_ok"]:
                test_results["issues"].append("验证脚本输出格式可能不正确")
                test_results["recommendations"].append("检查验证脚本是否输出正确的指标格式")
        
        # 检查输出目录
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        test_results["output_directory_exists"] = os.path.exists(output_dir)
        if test_results["output_directory_exists"]:
            # 查找模型文件
            model_files = list(Path(output_dir).glob("*.pth")) + list(Path(output_dir).glob("*.pt"))
            test_results["model_files_found"] = [str(f) for f in model_files]
            if not model_files:
                test_results["issues"].append("output 目录中未找到模型文件")
                test_results["recommendations"].append("先运行训练任务生成模型文件")
        else:
            test_results["issues"].append("output 目录不存在")
            test_results["recommendations"].append("确保 output 目录存在")
        
        # 检查配置文件
        configs_dir = os.path.join(os.path.dirname(__file__), "configs")
        if os.path.exists(configs_dir):
            config_files = list(Path(configs_dir).glob("*.yaml"))
            test_results["config_files_found"] = [f.stem for f in config_files]
            if not config_files:
                test_results["issues"].append("configs 目录中未找到配置文件")
                test_results["recommendations"].append("生成或复制配置文件到 configs 目录")
        else:
            test_results["issues"].append("configs 目录不存在")
            test_results["recommendations"].append("创建 configs 目录并添加配置文件")
        
        # 检查数据集
        dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
        if os.path.exists(dataset_dir):
            dataset_files = ["train.jsonl", "valid.jsonl", "test.jsonl"]
            missing_datasets = []
            for dataset_file in dataset_files:
                if not os.path.exists(os.path.join(dataset_dir, dataset_file)):
                    missing_datasets.append(dataset_file)
            
            if missing_datasets:
                test_results["issues"].append(f"缺少数据集文件: {', '.join(missing_datasets)}")
                test_results["recommendations"].append("确保所有必需的数据集文件存在")
        else:
            test_results["issues"].append("dataset 目录不存在")
            test_results["recommendations"].append("创建 dataset 目录并添加数据集文件")
        
        # 总体状态
        test_results["overall_status"] = "ready" if not test_results["issues"] else "needs_attention"
        
        return jsonify({
            "success": True,
            "test_results": test_results,
            "message": "验证环境测试完成" if not test_results["issues"] else f"发现 {len(test_results['issues'])} 个问题"
        })
        
    except Exception as e:
        logger.error(f"验证环境测试失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "test_results": None
        }), 500

def run_training(job: TrainingJob):
    """运行训练任务（仅训练，不包含验证）"""
    try:
        job.status = "running"
        job.start_time = datetime.now().isoformat()
        job.status_description = "正在初始化训练..."
        job.add_log(f"🚀 开始训练任务: {job.model_name}")
        
        # 在终端显示训练开始信息
        print(f"\n{'='*80}")
        print(f"🚀 vulcan 训练任务启动")
        print(f"{'='*80}")
        print(f"📋 任务详情:")
        print(f"   • 任务ID: {job.job_id}")
        print(f"   • 模型: {job.model_name}")
        print(f"   • 配置文件: {job.config_path}")
        print(f"   • 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 获取当前工作目录和项目根目录
        current_dir = os.path.dirname(__file__)
        project_root = current_dir
        
        job.add_log(f"📂 工作目录: {project_root}")
        print(f"📂 工作目录: {project_root}")
        
        # 构建训练命令 - 使用绝对路径
        train_script = os.path.join(project_root, "tools", "train.py")
        config_path = os.path.abspath(job.config_path)
        
        # 详细的文件存在性检查
        job.add_log(f"🔍 检查训练脚本: {train_script}")
        print(f"🔍 检查训练脚本: {train_script}")
        if not os.path.exists(train_script):
            error_msg = f"❌ 训练脚本不存在: {train_script}"
            job.add_log(error_msg)
            print(error_msg)
            job.status = "failed"
            job.status_description = "训练脚本不存在"
            return
            
        job.add_log(f"✅ 训练脚本存在")
        print(f"✅ 训练脚本存在")
        
        job.add_log(f"🔍 检查配置文件: {config_path}")
        print(f"🔍 检查配置文件: {config_path}")
        if not os.path.exists(config_path):
            error_msg = f"❌ 配置文件不存在: {config_path}"
            job.add_log(error_msg)
            print(error_msg)
            job.status = "failed"
            job.status_description = "配置文件不存在"
            return
            
        job.add_log(f"✅ 配置文件存在")
        print(f"✅ 配置文件存在")
        
        # 检查数据集目录
        dataset_dir = os.path.join(project_root, "dataset")
        job.add_log(f"🔍 检查数据集目录: {dataset_dir}")
        print(f"🔍 检查数据集目录: {dataset_dir}")
        if os.path.exists(dataset_dir):
            job.add_log(f"✅ 数据集目录存在")
            print(f"✅ 数据集目录存在")
        else:
            job.add_log(f"⚠️ 数据集目录不存在: {dataset_dir}")
            print(f"⚠️ 数据集目录不存在: {dataset_dir}")
        
        # 构建训练命令
        cmd = [
            sys.executable,
            train_script,
            "--cfg", config_path
        ]
        
        job.add_log(f"🔧 执行训练命令: {' '.join(cmd)}")
        print(f"🔧 执行训练命令: {' '.join(cmd)}")
        job.status_description = "正在启动训练进程..."
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # 确保输出不被缓冲
        env['PYTHONIOENCODING'] = 'utf-8'  # 确保中文输出正常
        
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=project_root,
            env=dict(os.environ, PYTHONUNBUFFERED='1')
        )
        
        job.process = process
        
        print(f"✅ 训练进程已启动，PID: {process.pid}")
        job.add_log(f"✅ 训练进程已启动，PID: {process.pid}")
        print(f"\n📋 训练实时输出:")
        print(f"{'='*60}")
        
        # 监控训练输出
        try:
            for line in process.stdout:
                line = line.strip()
                if line:
                    # 同时在终端和日志中显示
                    print(f"🚀 {line}")
                    job.add_log(line)
                    
                    # 解析训练进度
                    if 'epoch' in line.lower() or 'Epoch' in line:
                        try:
                            # 匹配多种Epoch格式: Epoch: [1/1], Epoch 1/10, epoch: 1
                            epoch_patterns = [
                                r'[Ee]poch[:\s]*\[(\d+)/(\d+)\]',  # Epoch: [1/1]
                                r'[Ee]poch[:\s]*(\d+)[/\s]*(\d+)',  # Epoch: 1/10 or Epoch 1 10
                                r'[Ee]poch[:\s]*(\d+)'  # Epoch: 1
                            ]
                            
                            for pattern in epoch_patterns:
                                epoch_match = re.search(pattern, line)
                                if epoch_match:
                                    current_epoch = int(epoch_match.group(1))
                                    if len(epoch_match.groups()) >= 2 and epoch_match.group(2):
                                        total_epochs = int(epoch_match.group(2))
                                        job.total_epochs = total_epochs
                                    job.current_epoch = current_epoch
                                    if job.total_epochs > 0:
                                        job.progress = (current_epoch / job.total_epochs) * 100
                                        print(f"📊 训练进度(Epoch): {current_epoch}/{job.total_epochs} ({job.progress:.1f}%)")
                                        job.add_log(f"训练进度(Epoch): {current_epoch}/{job.total_epochs} ({job.progress:.1f}%)")
                                    break
                        except Exception as e:
                            job.add_log(f"解析epoch信息失败: {e}")
                    
                    # 解析迭代进度 - 更精确的进度计算
                    if 'Iter:' in line or 'iter:' in line:
                        try:
                            # 匹配 Iter: [76/142] 格式
                            iter_match = re.search(r'[Ii]ter[:\s]*\[(\d+)/(\d+)\]', line)
                            if iter_match:
                                current_iter = int(iter_match.group(1))
                                total_iters = int(iter_match.group(2))
                                job.current_iteration = current_iter
                                job.total_iterations = total_iters
                                
                                # 使用迭代进度作为更精确的进度指示器
                                if total_iters > 0:
                                    iter_progress = (current_iter / total_iters) * 100
                                    # 如果有多个epoch，需要考虑当前epoch
                                    if job.total_epochs > 1 and job.current_epoch > 0:
                                        # 总进度 = (已完成的epoch + 当前epoch的进度) / 总epoch数
                                        total_progress = ((job.current_epoch - 1) + (current_iter / total_iters)) / job.total_epochs * 100
                                        job.progress = total_progress
                                    else:
                                        # 单epoch情况，直接使用迭代进度
                                        job.progress = iter_progress
                                    
                                    print(f"📈 迭代进度: {current_iter}/{total_iters} (总进度: {job.progress:.1f}%)")
                                    job.add_log(f"迭代进度: {current_iter}/{total_iters} (总进度: {job.progress:.1f}%)")
                        except Exception as e:
                            job.add_log(f"解析迭代信息失败: {e}")
                    
                    # 解析百分比进度条 - 作为备用进度指示器
                    if '%|' in line:
                        try:
                            # 匹配进度条中的百分比: 63%|██████▎
                            percent_match = re.search(r'(\d+)%\|', line)
                            if percent_match:
                                progress_percent = int(percent_match.group(1))
                                # 只有在没有其他进度信息时才使用这个
                                if job.progress == 0:
                                    job.progress = progress_percent
                                    print(f"📊 进度条进度: {progress_percent}%")
                                    job.add_log(f"进度条进度: {progress_percent}%")
                        except Exception as e:
                            job.add_log(f"解析进度条失败: {e}")
                    
                    # 解析验证指标 - 更加灵活的解析
                    if "Accuracy:" in line or "accuracy:" in line or "acc:" in line:
                        try:
                            # 支持多种格式: Accuracy: 0.85, accuracy: 85%, acc: 0.8500
                            patterns = [r'Accuracy[:\s]+([\d\.]+)', r'accuracy[:\s]+([\d\.]+)', r'acc[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                acc_match = re.search(pattern, line, re.IGNORECASE)
                                if acc_match:
                                    acc_value = float(acc_match.group(1))
                                    # 如果值大于1，可能是百分比格式
                                    if acc_value > 1:
                                        acc_value = acc_value / 100
                                    job.metrics["validation_accuracy"] = acc_value
                                    print(f"🎯 解析到验证准确率: {acc_value:.4f}")
                                    job.add_log(f"解析到验证准确率: {acc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析accuracy信息失败: {e} - 原始行: {line}")
                    
                    if "F1:" in line or "f1:" in line:
                        try:
                            patterns = [r'F1[:\s]+([\d\.]+)', r'f1[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                f1_match = re.search(pattern, line, re.IGNORECASE)
                                if f1_match:
                                    f1_value = float(f1_match.group(1))
                                    if f1_value > 1:
                                        f1_value = f1_value / 100
                                    job.metrics["validation_f1"] = f1_value
                                    print(f"📊 解析到验证F1: {f1_value:.4f}")
                                    job.add_log(f"解析到验证F1: {f1_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析F1信息失败: {e} - 原始行: {line}")
                    
                    if "Precision:" in line or "precision:" in line or "prec:" in line:
                        try:
                            patterns = [r'Precision[:\s]+([\d\.]+)', r'precision[:\s]+([\d\.]+)', r'prec[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                prec_match = re.search(pattern, line, re.IGNORECASE)
                                if prec_match:
                                    prec_value = float(prec_match.group(1))
                                    if prec_value > 1:
                                        prec_value = prec_value / 100
                                    job.metrics["validation_precision"] = prec_value
                                    print(f"📈 解析到验证精确率: {prec_value:.4f}")
                                    job.add_log(f"解析到验证精确率: {prec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析precision信息失败: {e} - 原始行: {line}")
                    
                    if "Recall:" in line or "recall:" in line or "rec:" in line:
                        try:
                            patterns = [r'Recall[:\s]+([\d\.]+)', r'recall[:\s]+([\d\.]+)', r'rec[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                rec_match = re.search(pattern, line, re.IGNORECASE)
                                if rec_match:
                                    rec_value = float(rec_match.group(1))
                                    if rec_value > 1:
                                        rec_value = rec_value / 100
                                    job.metrics["validation_recall"] = rec_value
                                    print(f"🔄 解析到验证召回率: {rec_value:.4f}")
                                    job.add_log(f"解析到验证召回率: {rec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析recall信息失败: {e} - 原始行: {line}")
                    
                    if "ROC_AUC:" in line or "AUC:" in line or "auc:" in line:
                        try:
                            patterns = [r'ROC_AUC[:\s]+([\d\.]+)', r'AUC[:\s]+([\d\.]+)', r'auc[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                auc_match = re.search(pattern, line, re.IGNORECASE)
                                if auc_match:
                                    auc_value = float(auc_match.group(1))
                                    if auc_value > 1:
                                        auc_value = auc_value / 100
                                    job.metrics["validation_auc"] = auc_value
                                    print(f"📌 解析到验证AUC: {auc_value:.4f}")
                                    job.add_log(f"解析到验证AUC: {auc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析AUC信息失败: {e} - 原始行: {line}")
                    
                    # 检测错误信息
                    if any(error_word in line.lower() for error_word in ['error', 'exception', 'failed', 'traceback']):
                        print(f"❌ 检测到错误信息: {line}")
                        job.add_log(f"检测到错误信息: {line}")
                    
                    # 解析训练损失
                    if "loss:" in line.lower() or "Loss:" in line:
                        try:
                            loss_match = re.search(r'[Ll]oss[:\s]+([\d\.]+)', line)
                            if loss_match:
                                loss_value = float(loss_match.group(1))
                                job.metrics["loss"] = loss_value
                                print(f"📉 当前损失: {loss_value:.4f}")
                                job.add_log(f"当前损失: {loss_value:.4f}")
                        except Exception as e:
                            job.add_log(f"解析loss信息失败: {e}")
                    
                    # 解析训练准确率（如果有的话）
                    if ("train" in line.lower() and "acc" in line.lower()) or ("training" in line.lower() and "accuracy" in line.lower()):
                        try:
                            acc_match = re.search(r'[Aa]ccuracy[:\s]+([\d\.]+)', line)
                            if acc_match:
                                acc_value = float(acc_match.group(1))
                                if acc_value > 1:  # 可能是百分比格式
                                    acc_value = acc_value / 100
                                job.metrics["accuracy"] = acc_value
                                print(f"🎯 训练准确率: {acc_value:.4f}")
                                job.add_log(f"训练准确率: {acc_value:.4f}")
                        except Exception as e:
                            job.add_log(f"解析训练accuracy信息失败: {e}")
        
        except Exception as e:
            error_msg = f"读取训练输出时发生错误: {e}"
            job.add_log(error_msg)
            print(f"❌ {error_msg}")
        
        print(f"{'='*60}")
        print(f"📊 等待训练进程完成...")
        
        # 等待进程结束
        return_code = process.wait()
        
        print(f"📊 训练进程结束，返回码: {return_code}")
        
        # 判断训练是否成功
        if return_code == 0:
            job.status = "completed"  # 暂时标记为完成，稍后会改为验证中
            job.add_log("🎉 训练阶段完成！")
            print(f"✅ 训练阶段完成！")
            
            # 确保训练进度为100%
            job.progress = 100
            
            # 显示详细的训练完成摘要
            print(f"\n🎯 训练阶段完成摘要:")
            print(f"{'='*60}")
            job.add_log("🎯 训练阶段完成摘要:")
            job.add_log("="*60)
            
            # 基本信息
            basic_info = f"📋 训练基本信息:\n• 模型: {job.model_name}\n• 配置文件: {job.config_path}\n• 训练轮数: {job.current_epoch}/{job.total_epochs}"
            if job.total_iterations > 0:
                basic_info += f"\n• 总迭代数: {job.current_iteration}/{job.total_iterations}"
            basic_info += f"\n• 训练用时: {job.get_duration()}"
            
            print(basic_info)
            job.add_log(basic_info)
            
            # 显示最终训练指标
            if job.metrics:
                metrics_info = "\n📊 最终训练指标:"
                print(metrics_info)
                job.add_log(metrics_info)
                
                for key, value in job.metrics.items():
                    if not key.startswith('validation_'):  # 只显示训练指标
                        if isinstance(value, (int, float)):
                            metric_line = f"• {key}: {value:.4f}"
                        else:
                            metric_line = f"• {key}: {value}"
                        print(f"  {metric_line}")
                        job.add_log(f"  {metric_line}")
            
            # 模型保存信息
            output_dir = os.path.join(project_root, "output")
            if os.path.exists(output_dir):
                model_files = list(Path(output_dir).glob("*.pth")) + list(Path(output_dir).glob("*.pt"))
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    model_info = f"\n💾 模型保存信息:\n• 输出目录: {output_dir}\n• 最新模型: {latest_model.name}\n• 模型大小: {latest_model.stat().st_size / 1024 / 1024:.2f} MB"
                    print(model_info)
                    job.add_log(model_info)
                    
                    # ====== 检查是否需要自动验证 ======
                    if job.auto_validation:
                        print(f"\n🔬 开始自动验证模型...")
                        job.add_log("\n🔬 开始自动验证模型...")
                        job.status = "running"  # 继续运行状态，进入验证阶段
                        job.status_description = "正在验证模型..."
                        
                        # 运行验证并将结果合并到同一个任务中
                        run_validation_in_training(job, project_root)
                    else:
                        print(f"\n✅ 训练完成，未启用自动验证")
                        job.add_log("\n✅ 训练完成，未启用自动验证")
                        job.status = "completed"
                        job.status_description = "训练完成"
                    
                else:
                    no_model_info = f"\n⚠️ 在输出目录中未找到保存的模型文件"
                    if job.auto_validation:
                        no_model_info += "，跳过验证"
                    print(no_model_info)
                    job.add_log(no_model_info)
                    job.status = "completed"
            else:
                validation_skip_info = f"\n⚠️ 输出目录不存在"
                if job.auto_validation:
                    validation_skip_info += "，跳过验证"
                print(validation_skip_info)
                job.add_log(validation_skip_info)
                job.status = "completed"
                
        else:
            job.status = "failed"
            failure_msg = f"❌ 训练失败，返回码: {return_code}"
            job.add_log(failure_msg)
            print(failure_msg)
            
            # 显示详细的失败信息
            print(f"\n💥 训练失败详情:")
            print(f"{'='*60}")
            job.add_log("💥 训练失败详情:")
            job.add_log("="*60)
            
            failure_details = f"📋 失败信息:\n• 模型: {job.model_name}\n• 配置文件: {job.config_path}\n• 失败时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n• 返回码: {return_code}"
            print(failure_details)
            job.add_log(failure_details)
            
            # 添加更详细的错误信息和故障排除建议
            troubleshooting = f"\n🔧 故障排除建议:\n• 1. 检查数据集路径是否正确\n• 2. 确认GPU/CUDA是否可用\n• 3. 验证系统内存是否充足\n• 4. 检查依赖库是否安装正确\n• 5. 查看详细日志文件: {job.get_log_file_path()}\n• 6. 尝试降低batch_size或学习率\n• 7. 检查配置文件格式是否正确"
            print(troubleshooting)
            job.add_log(troubleshooting)
            
            print(f"{'='*60}")
            job.add_log("="*60)
        
        job.end_time = datetime.now().isoformat()
        duration_msg = f"⏱️ 训练用时: {job.get_duration()}"
        job.add_log(duration_msg)
        print(f"{duration_msg}")
        
        print(f"{'='*80}")
        print(f"🚀 vulcan 训练任务完成")
        print(f"{'='*80}\n")
        
        # 确保日志保存到文件
        job.save_logs_to_file()
        
    except Exception as e:
        job.status = "failed"
        job.status_description = f"训练异常: {str(e)}"
        job.add_log(f"❌ 训练异常: {str(e)}")
        job.end_time = datetime.now().isoformat()
        logger.error(f"训练任务 {job.job_id} 异常: {e}", exc_info=True)
        print(f"💥 [训练异常] {str(e)}")
        
        # 添加详细的异常信息到日志
        import traceback
        job.add_log("📝 详细异常信息:")
        job.add_log(traceback.format_exc())
        
        print(f"📝 详细异常信息:")
        print(traceback.format_exc())
        
        # 确保日志保存到文件
        job.save_logs_to_file()

def run_validation_in_training(job: TrainingJob, project_root: str):
    """在训练任务中运行验证，将验证结果合并到训练任务中"""
    try:
        # 设置验证阶段
        job.current_phase = "validation"
        job.status_description = "正在验证模型..."
        
        # 验证阶段开始
        job.add_log("\n" + "="*80)
        job.add_log("🔬 vulcan 自动验证阶段启动")
        job.add_log("="*80)
        print(f"\n{'='*80}")
        print(f"🔬 vulcan 自动验证阶段启动")
        print(f"{'='*80}")
        
        # 构建验证命令
        val_script = os.path.join(project_root, "tools", "val.py")
        config_path = os.path.abspath(job.config_path)
        
        # 验证文件存在性
        if not os.path.exists(val_script):
            error_msg = f"❌ 验证脚本不存在: {val_script}"
            job.add_log(error_msg)
            print(error_msg)
            job.current_phase = "completed"  # 跳过验证，直接完成
            return
        
        job.add_log(f"✅ 验证脚本: {val_script}")
        print(f"✅ 验证脚本: {val_script}")
        job.add_log(f"✅ 配置文件: {config_path}")
        print(f"✅ 配置文件: {config_path}")
        
        # 构建验证命令
        cmd = [sys.executable, val_script, "--cfg", config_path]
        
        job.add_log(f"🔧 执行验证命令: {' '.join(cmd)}")
        print(f"🔧 执行验证命令: {' '.join(cmd)}")
        
        # 启动验证进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=project_root,
            env=dict(os.environ, PYTHONUNBUFFERED='1')
        )
        
        job.add_log(f"✅ 验证进程已启动，PID: {process.pid}")
        print(f"✅ 验证进程已启动，PID: {process.pid}")
        print(f"\n📋 验证实时输出:")
        print(f"{'='*60}")
        
        # 存储验证指标的临时变量
        validation_metrics = {}
        
        # 监控验证输出
        try:
            for line in process.stdout:
                line = line.strip()
                if line:
                    # 同时在终端和日志中显示
                    print(f"🔬 {line}")
                    job.add_log(line)
                    
                    # 解析验证进度
                    if '%' in line and ('|' in line or '/' in line):
                        try:
                            # 解析百分比进度
                            percent_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                            if percent_match:
                                progress_percent = float(percent_match.group(1))
                                # 验证进度从50%开始到100%（训练占50%，验证占50%）
                                job.progress = 50 + (progress_percent / 2)
                                print(f"📊 总体进度: {job.progress:.1f}% (验证: {progress_percent:.1f}%)")
                                job.add_log(f"验证进度: {progress_percent:.1f}%")
                        except Exception as e:
                            job.add_log(f"解析验证进度失败: {e}")
                    
                    # 解析验证指标
                    if "Accuracy:" in line or "accuracy:" in line or "acc:" in line:
                        try:
                            patterns = [r'Accuracy[:\s]+([\d\.]+)', r'accuracy[:\s]+([\d\.]+)', r'acc[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                acc_match = re.search(pattern, line, re.IGNORECASE)
                                if acc_match:
                                    acc_value = float(acc_match.group(1))
                                    if acc_value > 1:
                                        acc_value = acc_value / 100
                                    validation_metrics["validation_accuracy"] = acc_value
                                    job.metrics["validation_accuracy"] = acc_value
                                    print(f"🎯 验证准确率: {acc_value:.4f}")
                                    job.add_log(f"解析到验证准确率: {acc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析accuracy信息失败: {e}")
                    
                    if "F1:" in line or "f1:" in line:
                        try:
                            patterns = [r'F1[:\s]+([\d\.]+)', r'f1[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                f1_match = re.search(pattern, line, re.IGNORECASE)
                                if f1_match:
                                    f1_value = float(f1_match.group(1))
                                    if f1_value > 1:
                                        f1_value = f1_value / 100
                                    validation_metrics["validation_f1"] = f1_value
                                    job.metrics["validation_f1"] = f1_value
                                    print(f"📊 验证F1: {f1_value:.4f}")
                                    job.add_log(f"解析到验证F1: {f1_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析F1信息失败: {e}")
                    
                    if "Precision:" in line or "precision:" in line or "prec:" in line:
                        try:
                            patterns = [r'Precision[:\s]+([\d\.]+)', r'precision[:\s]+([\d\.]+)', r'prec[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                prec_match = re.search(pattern, line, re.IGNORECASE)
                                if prec_match:
                                    prec_value = float(prec_match.group(1))
                                    if prec_value > 1:
                                        prec_value = prec_value / 100
                                    validation_metrics["validation_precision"] = prec_value
                                    job.metrics["validation_precision"] = prec_value
                                    print(f"📈 验证精确率: {prec_value:.4f}")
                                    job.add_log(f"解析到验证精确率: {prec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析precision信息失败: {e}")
                    
                    if "Recall:" in line or "recall:" in line or "rec:" in line:
                        try:
                            patterns = [r'Recall[:\s]+([\d\.]+)', r'recall[:\s]+([\d\.]+)', r'rec[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                rec_match = re.search(pattern, line, re.IGNORECASE)
                                if rec_match:
                                    rec_value = float(rec_match.group(1))
                                    if rec_value > 1:
                                        rec_value = rec_value / 100
                                    validation_metrics["validation_recall"] = rec_value
                                    job.metrics["validation_recall"] = rec_value
                                    print(f"🔄 验证召回率: {rec_value:.4f}")
                                    job.add_log(f"解析到验证召回率: {rec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析recall信息失败: {e}")
                    
                    if "ROC_AUC:" in line or "AUC:" in line or "auc:" in line:
                        try:
                            patterns = [r'ROC_AUC[:\s]+([\d\.]+)', r'AUC[:\s]+([\d\.]+)', r'auc[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                auc_match = re.search(pattern, line, re.IGNORECASE)
                                if auc_match:
                                    auc_value = float(auc_match.group(1))
                                    if auc_value > 1:
                                        auc_value = auc_value / 100
                                    validation_metrics["validation_auc"] = auc_value
                                    job.metrics["validation_auc"] = auc_value
                                    print(f"📌 验证AUC: {auc_value:.4f}")
                                    job.add_log(f"解析到验证AUC: {auc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析AUC信息失败: {e}")
                    
                    if "PR_AUC:" in line or "pr_auc:" in line:
                        try:
                            patterns = [r'PR_AUC[:\s]+([\d\.]+)', r'pr_auc[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                pr_auc_match = re.search(pattern, line, re.IGNORECASE)
                                if pr_auc_match:
                                    pr_auc_value = float(pr_auc_match.group(1))
                                    if pr_auc_value > 1:
                                        pr_auc_value = pr_auc_value / 100
                                    validation_metrics["validation_pr_auc"] = pr_auc_value
                                    job.metrics["validation_pr_auc"] = pr_auc_value
                                    print(f"📊 验证PR_AUC: {pr_auc_value:.4f}")
                                    job.add_log(f"解析到验证PR_AUC: {pr_auc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"解析PR_AUC信息失败: {e}")
                    
                    # 检测验证结果表格（tabulate输出）
                    if "Class" in line and "F1" in line and "Acc" in line:
                        job.add_log("📊 验证结果表格:")
                        print("📊 验证结果表格:")
                    
                    # 如果包含分隔符或表格数据，也记录
                    if "---" in line or ("|" in line and any(word in line for word in ["F1", "Acc", "Prec", "Rec"])):
                        # 这些是表格的一部分，确保被记录
                        pass
                    
                    # 检测错误信息
                    if any(error_word in line.lower() for error_word in ['error', 'exception', 'failed', 'traceback']):
                        print(f"❌ 检测到错误信息: {line}")
                        job.add_log(f"检测到错误信息: {line}")
        
        except Exception as e:
            error_msg = f"读取验证输出时发生错误: {e}"
            job.add_log(error_msg)
            print(f"❌ {error_msg}")
        
        print(f"{'='*60}")
        print(f"📊 等待验证进程完成...")
        
        # 等待进程结束
        return_code = process.wait()
        print(f"📊 验证进程结束，返回码: {return_code}")
        
        # 确保进度为100%
        job.progress = 100
        
        # 判断验证是否成功
        has_metrics = any(key.startswith('validation_') for key in job.metrics.keys())
        
        if return_code == 0 or has_metrics:
            job.status = "completed"
            job.current_phase = "completed"  # 标记为完全完成
            job.add_log("✅ 验证完成！")
            print(f"✅ 验证完成！")
            
            # 显示完整的训练+验证结果摘要
            print(f"\n🎯 完整训练+验证结果摘要:")
            print(f"{'='*80}")
            job.add_log("\n🎯 完整训练+验证结果摘要:")
            job.add_log("="*80)
            
            # 显示验证指标
            if validation_metrics:
                validation_info = "\n📊 验证结果:"
                print(validation_info)
                job.add_log(validation_info)
                
                for key, value in validation_metrics.items():
                    if isinstance(value, (int, float)):
                        metric_line = f"• {key.replace('validation_', '')}: {value:.4f}"
                        if value <= 1.0:  # 添加百分比显示
                            metric_line += f" ({value*100:.1f}%)"
                    else:
                        metric_line = f"• {key.replace('validation_', '')}: {value}"
                    print(f"  {metric_line}")
                    job.add_log(f"  {metric_line}")
            
            # 性能评估
            if validation_metrics.get("validation_accuracy", 0) > 0:
                acc = validation_metrics["validation_accuracy"]
                assessment = "\n🏆 性能评估:"
                if acc > 0.9:
                    assessment += "\n  🎉 优秀！模型性能超过90%，可以直接部署使用"
                elif acc > 0.8:
                    assessment += "\n  👍 良好！模型性能超过80%，建议进一步测试"
                elif acc > 0.7:
                    assessment += "\n  ⚠️ 一般！模型性能在70-80%之间，建议优化"
                else:
                    assessment += "\n  ❌ 较差！模型性能低于70%，需要重新训练"
                
                print(assessment)
                job.add_log(assessment)
            
            # 后续操作建议
            next_steps = f"\n🚀 后续操作建议:\n• 1. 查看完整日志: {job.get_log_file_path()}\n• 2. 使用数据集优化功能进一步提升效果\n• 3. 部署模型进行实际应用测试\n• 4. 对比不同配置的训练结果"
            print(next_steps)
            job.add_log(next_steps)
            
            print(f"{'='*80}")
            job.add_log("="*80)
            
        else:
            job.status = "failed"
            job.current_phase = "completed"  # 即使失败也标记为完成
            failure_msg = f"❌ 验证失败，返回码: {return_code}"
            job.add_log(failure_msg)
            print(failure_msg)
        
        print(f"🔬 vulcan 训练+验证任务完成")
        print(f"{'='*80}\n")
        
    except Exception as e:
        job.add_log(f"❌ 验证阶段异常: {str(e)}")
        print(f"❌ 验证阶段异常: {str(e)}")
        logger.error(f"验证阶段异常: {e}")
        job.current_phase = "completed"  # 即使验证失败，训练仍然成功
        job.status = "completed"  # 即使验证失败，训练仍然成功

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 vulcan-Detection Backend Server")
    print("=" * 60)
    print("📋 可用API接口:")
    print("  GET  /api/health                    - 健康检查")
    print("  GET  /api/models                    - 获取可用模型")
    print("  GET  /api/datasets                  - 获取可用数据集")
    print("  POST /api/generate-config           - 生成配置文件")
    print("  POST /api/generate-config-from-template - 基于模板生成配置")
    print("  POST /api/start-training            - 启动训练")
    print("  GET  /api/training-status/<job_id>  - 获取训练状态")
    print("  GET  /api/training-logs/<job_id>    - 获取训练日志")
    print("  POST /api/start-validation/<config_name> - 启动验证")
    print("  GET  /api/validation-status/<job_id> - 获取验证状态")
    print("  GET  /api/test-validation          - 测试验证环境和配置")
    print("  POST /api/start-dataset-optimization - 启动数据集优化")
    print("  GET  /api/optimization-status/<job_id> - 获取优化状态")
    print("  GET  /api/optimization-logs/<job_id> - 获取优化日志")
    print("  POST /api/start-paper-search       - 启动论文检索")
    print("  GET  /api/paper-search-status/<job_id> - 获取论文检索状态")
    print("  GET  /api/paper-search-logs/<job_id> - 获取论文检索日志")
    print("  GET  /api/papers/<job_id>          - 获取检索到的论文")
    print("  POST /api/start-data-collection    - 启动数据收集任务")
    print("  GET  /api/data-collection-status/<job_id> - 获取数据收集状态")
    print("  GET  /api/data-collection-logs/<job_id> - 获取数据收集日志")
    print("  GET  /api/data-collection-results/<job_id> - 获取数据收集结果")
    print("=" * 60)
    
    # 设置日志级别以减少警告信息
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # 确保输出不被缓冲
    import sys
    import os
    
    # 设置环境变量确保输出不被缓冲
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # 重新配置标准输出
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print("✅ 输出缓冲已禁用，实时输出已启用")
    print("🚀 启动Flask应用...")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False) 