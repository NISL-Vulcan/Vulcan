#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vulcan-Detection Backend API Server
EN
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

# EN
from vulcan.framework.config_templates import ConfigTemplateManager

app = Flask(__name__)
CORS(app)  # EN

# EN
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EN
training_jobs = {}
training_configs = {}
optimization_jobs = {}  # EN:EN
jobs = {}  # EN:EN(EN)

class TrainingJob:
    """EN"""
    def __init__(self, job_id: str, config_path: str, model_name: str):
        self.job_id = job_id
        self.config_path = config_path
        self.model_name = model_name
        self.status = "pending"  # pending, running, completed, failed
        self.status_description = "EN..."  # EN
        self.progress = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_iteration = 0  # EN
        self.total_iterations = 0   # EN
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
        # EN
        self.log_file_path = f"logs/training_{job_id}.log"
        # EN - EN
        self._validation_started = False
        # EN
        self.auto_validation = False  # EN,EN
        # EN
        self.current_phase = "training"  # training, validation, completed
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """EN"""
        import os
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
            logger.error(f"EN: {e}")
        
        # EN,EN
        if len(self.logs) > 2000:  # EN
            self.logs = self.logs[-1500:]  # EN
    
    def get_logs(self, limit: int = None) -> List[str]:
        """EN,EN"""
        try:
            # EN
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    file_logs = f.readlines()
                    # EN
                    file_logs = [log.strip() for log in file_logs if log.strip()]
                    
                    if limit:
                        return file_logs[-limit:]
                    return file_logs
        except Exception as e:
            logger.error(f"EN: {e}")
        
        # EN,EN
        if limit:
            return self.logs[-limit:]
        return self.logs.copy()
    
    def get_recent_logs(self, count: int = 50) -> List[str]:
        """EN"""
        return self.get_logs(limit=count)
    
    def get_full_logs(self) -> List[str]:
        """EN"""
        return self.get_logs()
    
    def save_logs_to_file(self):
        """EN"""
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                for log in self.logs:
                    f.write(log + '\n')
        except Exception as e:
            logger.error(f"EN: {e}")
    
    def load_logs_from_file(self):
        """EN"""
        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    file_logs = f.readlines()
                    self.logs = [log.strip() for log in file_logs if log.strip()]
        except Exception as e:
            logger.error(f"EN: {e}")

    def get_duration(self) -> str:
        """EN"""
        if self.start_time and self.end_time:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(self.end_time)
            duration = end_dt - start_dt
            return str(duration)
        return "N/A"
    
    def get_log_file_path(self) -> str:
        """EN"""
        return self.log_file_path

class ConfigGenerator:
    """EN"""
    
    def __init__(self, configs_dir: str = "generated_configs"):
        self.configs_dir = Path(configs_dir)
        self.configs_dir.mkdir(exist_ok=True)  # EN
        self.template_manager = ConfigTemplateManager()
        self.legacy_templates = self._load_legacy_templates()
    
    def _load_legacy_templates(self) -> Dict[str, Dict]:
        """EN"""
        templates = {}
        # ENconfigsEN
        config_root = Path("configs")
        if config_root.exists():
            for config_file in config_root.glob("*.yaml"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        templates[config_file.stem] = config
                except Exception as e:
                    logger.error(f"EN {config_file}: {e}")
        return templates
    
    def get_available_models(self) -> List[str]:
        """EN"""
        # EN
        template_models = self.template_manager.list_models()
        
        # ENlegacyEN
        legacy_models = set()
        for template in self.legacy_templates.values():
            if 'MODEL' in template and 'NAME' in template['MODEL']:
                legacy_models.add(template['MODEL']['NAME'])
        
        # EN
        all_models = list(set(template_models + list(legacy_models)))
        return all_models
    
    def get_available_datasets(self) -> List[str]:
        """EN"""
        datasets = set()
        
        # EN
        for template_name in self.template_manager.list_templates():
            template = self.template_manager.get_template(template_name)
            if template and template.template.get('DATASET', {}).get('NAME'):
                datasets.add(template.template['DATASET']['NAME'])
        
        # ENlegacyEN
        for template in self.legacy_templates.values():
            if 'DATASET' in template and 'NAME' in template['DATASET']:
                datasets.add(template['DATASET']['NAME'])
        
        return list(datasets)
    
    def generate_config(self, 
                       model_name: str, 
                       dataset_name: str, 
                       training_params: Dict[str, Any],
                       device: str = "cuda") -> Dict[str, Any]:
        """EN"""
        try:
            # EN
            required_params = ['epochs', 'batch_size', 'learning_rate']
            missing_params = [param for param in required_params if param not in training_params or training_params[param] is None]
            
            if missing_params:
                raise ValueError(f"EN: {', '.join(missing_params)}")
            
            # EN
            if training_params['epochs'] <= 0:
                raise ValueError("epochs EN 0")
            if training_params['batch_size'] <= 0:
                raise ValueError("batch_size EN 0")
            if training_params['learning_rate'] <= 0:
                raise ValueError("learning_rate EN 0")
            
            # EN
            template = self.template_manager.get_template_by_model(model_name)
            if template:
                logger.info(f"EN {model_name} EN")
                
                # EN,EN
                config_params = {
                    'DEVICE': device,
                    'TRAIN.BATCH_SIZE': training_params['batch_size'],
                    'TRAIN.EPOCHS': training_params['epochs'],
                    'TRAIN.EVAL_INTERVAL': training_params.get('eval_interval', 1),
                    'OPTIMIZER.LR': training_params['learning_rate'],
                    'SAVE_DIR': training_params.get('save_dir', 'output'),
                    'DATASET.NAME': dataset_name
                }
                
                # EN
                if training_params.get('data_path'):
                    config_params['DATASET.ROOT'] = training_params['data_path']
                
                config = template.generate_config(**config_params)
                return config
            
            # EN,ENlegacyEN
            logger.info(f"ENlegacyEN {model_name} EN")
            best_template = self._find_best_legacy_template(model_name, dataset_name)
            
            if best_template:
                config = self._deep_copy_dict(best_template)
                
                # EN,EN
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
            
            # EN,EN
            logger.warning(f"EN {model_name} EN,EN")
            return self._generate_basic_config(model_name, dataset_name, training_params, device)
            
        except Exception as e:
            logger.error(f"EN: {e}")
            # EN
            return self._generate_basic_config(model_name, dataset_name, training_params, device)
    
    def _generate_basic_config(self, model_name: str, dataset_name: str, 
                              training_params: Dict[str, Any], device: str) -> Dict[str, Any]:
        """EN"""
        # EN
        if 'epochs' not in training_params or training_params['epochs'] <= 0:
            raise ValueError("epochs EN 0")
        if 'batch_size' not in training_params or training_params['batch_size'] <= 0:
            raise ValueError("batch_size EN 0")
        if 'learning_rate' not in training_params or training_params['learning_rate'] <= 0:
            raise ValueError("learning_rate EN 0")
        
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
        """ENlegacyEN"""
        best_match = None
        best_score = 0
        
        for template_name, template in self.legacy_templates.items():
            score = 0
            
            # EN
            if 'MODEL' in template and 'NAME' in template['MODEL']:
                template_model = template['MODEL']['NAME'].lower()
                if model_name.lower() in template_model or template_model in model_name.lower():
                    score += 10
            
            # EN
            if 'DATASET' in template and 'NAME' in template['DATASET']:
                template_dataset = template['DATASET']['NAME'].lower()
                if dataset_name.lower() in template_dataset or template_dataset in dataset_name.lower():
                    score += 5
            
            if score > best_score:
                best_score = score
                best_match = template
        
        # EN,EN
        if best_match is None and self.legacy_templates:
            best_match = list(self.legacy_templates.values())[0]
        
        return best_match
    
    def _deep_copy_dict(self, d: Dict) -> Dict:
        """EN"""
        import copy
        return copy.deepcopy(d)
    
    def save_config(self, config: Dict[str, Any], filename: str) -> str:
        """EN"""
        output_dir = Path("generated_configs")
        output_dir.mkdir(exist_ok=True)
        
        config_path = output_dir / f"{filename}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        return str(config_path)

# EN
config_generator = ConfigGenerator()

@app.route('/api/models', methods=['GET'])
def get_models():
    """EN"""
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
    """EN"""
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
    """EN"""
    try:
        data = request.json
        
        # EN
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
        
        # EN
        config = config_generator.generate_config(
            model_name=model_name,
            dataset_name=dataset_name,
            training_params=training_params,
            device=device
        )
        
        # ENID
        config_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset_name}_{timestamp}"
        
        # EN
        config_path = config_generator.save_config(config, filename)
        
        # EN
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
            "message": f"EN: {filename}.yaml"
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/generate-config-from-template', methods=['POST'])
def generate_config_from_template():
    """EN regvd_reveal.yaml EN"""
    try:
        data = request.json
        
        # EN,EN regvd_reveal EN
        model_name = data.get('model_name', 'GNNReGVD')
        dataset_name = data.get('dataset_name', 'ReGVD')
        device = data.get('device', 'cuda')
        batch_size = data.get('batch_size', 128)
        epochs = data.get('epochs', 2)
        learning_rate = data.get('learning_rate', 0.001)
        save_dir = data.get('save_dir', 'output')
        
        # EN regvd_reveal.yaml EN
        template_path = os.path.join('configs', 'regvd_reveal.yaml')
        if not os.path.exists(template_path):
            return jsonify({
                "success": False,
                "error": f"EN regvd_reveal.yaml EN configs EN"
            }), 404
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # EN
        config_content = template_content.replace('cuda', device)
        config_content = config_content.replace('output', save_dir)
        config_content = config_content.replace('GNNReGVD', model_name)
        config_content = config_content.replace('ReGVD', dataset_name)
        config_content = config_content.replace('128', str(batch_size))
        config_content = config_content.replace('2', str(epochs))
        config_content = config_content.replace('0.001', str(learning_rate))
        
        # EN
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = f"regvd_reveal_{model_name}_{timestamp}"
        
        # EN configs EN
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)
        config_path = configs_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # EN
        try:
            config_data = yaml.safe_load(config_content)
        except:
            config_data = {}
        
        logger.info(f"EN: {config_path}")
        
        return jsonify({
            "success": True,
            "config_name": config_name,
            "config_path": str(config_path),
            "config_content": config_content,
            "config_data": config_data,
            "template_used": "regvd_reveal",
            "message": f"EN regvd_reveal.yaml EN: {config_name}.yaml"
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/start-training', methods=['POST'])
def start_training():
    """EN"""
    try:
        data = request.json
        config_id = data.get('config_id')
        auto_validation = data.get('auto_validation', False)  # EN
        
        if not config_id or config_id not in training_configs:
            return jsonify({
                "success": False,
                "error": "ENID"
            }), 400
        
        config_info = training_configs[config_id]
        job_id = str(uuid.uuid4())
        
        # EN
        job = TrainingJob(
            job_id=job_id,
            config_path=config_info['config_path'],
            model_name=config_info['model_name']
        )
        
        # EN
        job.auto_validation = auto_validation
        
        # EN
        config = config_info['config']
        job.total_epochs = config.get('TRAIN', {}).get('EPOCHS', 10)
        
        training_jobs[job_id] = job
        
        # EN
        training_thread = threading.Thread(
            target=run_training,
            args=(job,),
            daemon=True
        )
        training_thread.start()
        
        validation_message = ",EN" if auto_validation else ""
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "auto_validation": auto_validation,
            "message": f"EN: {job.model_name}{validation_message}"
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/training-status/<job_id>', methods=['GET'])
def get_training_status(job_id: str):
    """EN"""
    try:
        print(f" [API] EN: {job_id}")
        
        if job_id not in training_jobs:
            error_msg = "EN"
            print(f" {error_msg}: {job_id}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        job = training_jobs[job_id]
        
        print(f" EN: {job.status}, EN: {job.progress}%, EN: {job.current_epoch}/{job.total_epochs}")
        
        # EN
        is_validation_phase = job.metrics and (
            job.metrics.get('validation_accuracy', 0) > 0 or
            job.metrics.get('validation_f1', 0) > 0 or
            job.metrics.get('validation_precision', 0) > 0 or
            job.metrics.get('validation_recall', 0) > 0 or
            job.metrics.get('validation_loss', 0) > 0 or
            job.metrics.get('validation_auc', 0) > 0
        )
        
        # EN
        training_metrics = {}
        validation_metrics = {}
        
        for key, value in job.metrics.items():
            if key.startswith('validation_'):
                validation_metrics[key] = value
            else:
                training_metrics[key] = value
        
        if training_metrics:
            print(f" EN: {training_metrics}")
        if validation_metrics:
            print(f" EN: {validation_metrics}")
        
        # EN
        if job.status == "completed":
            # EN,EN
            logs = job.get_full_logs()
            log_count = len(logs)
            print(f" EN,EN({log_count}EN)")
        elif job.status == "failed":
            # EN,EN
            logs = job.get_full_logs()
            log_count = len(logs)
            print(f" EN,EN({log_count}EN)")
        else:
            # EN,EN
            logs = job.get_recent_logs(100)  # EN
            log_count = len(logs)
            print(f" EN,EN({log_count}EN)")
        
        # EN
        response_data = {
            "success": True,
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "current_iteration": job.current_iteration,  # EN
            "total_iterations": job.total_iterations,    # EN
            "start_time": job.start_time,
            "end_time": job.end_time,
            "logs": logs,
            "log_count": log_count,  # EN
            "log_file_path": job.get_log_file_path(),  # EN
            
            # EN
            "training_metrics": training_metrics,
            
            # EN
            "validation_metrics": validation_metrics,
            "is_validation_phase": is_validation_phase,
            
            # EN:EN metrics EN
            "metrics": job.metrics,
            
            # EN
            "model_name": job.model_name,
            "config_path": job.config_path,
            
            # EN
            "status_description": get_status_description(job.status, is_validation_phase),
            
            # EN
            "performance_summary": generate_performance_summary(training_metrics, validation_metrics),
            
            # EN:EN
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
        
        # EN,EN
        if job.status == "completed":
            response_data["final_results"] = generate_final_results_summary(training_metrics, validation_metrics, job)
            # EN
            response_data["completion_summary"] = {
                "total_logs": log_count,
                "training_duration": job.get_duration(),
                "final_logs_preview": logs[-20:] if len(logs) > 20 else logs,  # EN20EN
                "log_file_available": os.path.exists(job.get_log_file_path())
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"EN: {e}"
        logger.error(error_msg)
        print(f" {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def get_status_description(status: str, is_validation_phase: bool) -> str:
    """EN"""
    if status == "pending":
        return "EN"
    elif status == "running":
        if is_validation_phase:
            return "EN"
        else:
            return "EN"
    elif status == "completed":
        return "EN"
    elif status == "failed":
        return "EN"
    else:
        return "EN"

def generate_performance_summary(training_metrics: dict, validation_metrics: dict) -> dict:
    """EN"""
    summary = {
        "training": {},
        "validation": {},
        "comparison": {}
    }
    
    # EN
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
    
    # EN
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
    
    # ENvsEN
    if "accuracy" in training_metrics and "validation_accuracy" in validation_metrics:
        diff = validation_metrics["validation_accuracy"] - training_metrics["accuracy"]
        summary["comparison"]["accuracy_diff"] = {
            "value": diff,
            "percentage": diff * 100,
            "formatted": f"{diff:.4f} ({diff*100:+.1f}%)",
            "interpretation": "EN" if diff > 0 else "EN" if diff < -0.05 else "EN"
        }
    
    return summary

def generate_final_results_summary(training_metrics: dict, validation_metrics: dict, job) -> dict:
    """EN"""
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
    
    # EN
    if validation_metrics:
        # EN
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
        # EN,EN
        if "accuracy" in training_metrics:
            final_results["key_metrics"]["final_accuracy"] = {
                "value": training_metrics["accuracy"],
                "formatted": f"{training_metrics['accuracy']:.4f} ({training_metrics['accuracy']*100:.1f}%)",
                "source": "training"
            }
    
    # EN
    if validation_metrics:
        acc = validation_metrics.get("validation_accuracy", 0)
        f1 = validation_metrics.get("validation_f1", 0)
        
        if acc > 0.9:
            final_results["recommendations"].append(" EN!EN.")
        elif acc > 0.8:
            final_results["recommendations"].append(" EN,EN.")
        elif acc > 0.7:
            final_results["recommendations"].append(" EN,EN.")
        else:
            final_results["recommendations"].append(" EN,EN.")
        
        if f1 > 0 and abs(acc - f1) > 0.1:
            final_results["recommendations"].append(" ENF1EN,EN.")
    
    # ENvsEN
    if training_metrics and validation_metrics:
        train_acc = training_metrics.get("accuracy", 0)
        val_acc = validation_metrics.get("validation_accuracy", 0)
        
        if train_acc > 0 and val_acc > 0:
            diff = train_acc - val_acc
            if diff > 0.1:
                final_results["recommendations"].append(" EN,EN.")
            elif diff < -0.05:
                final_results["recommendations"].append(" EN,EN.")
    
    return final_results

@app.route('/api/training-logs/<job_id>', methods=['GET'])
def get_training_logs(job_id: str):
    """EN"""
    try:
        if job_id not in training_jobs:
            return jsonify({
                "success": False,
                "error": "EN"
            }), 404
        
        job = training_jobs[job_id]
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "logs": job.get_full_logs()
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ENAPIEN
@app.route('/api/get-config/<config_name>', methods=['GET'])
def get_existing_config(config_name: str):
    """EN"""
    try:
        # EN
        config_path = os.path.join('configs', f'{config_name}.yaml')
        
        if not os.path.exists(config_path):
            return jsonify({
                "success": False,
                "error": f"EN {config_name}.yaml EN"
            }), 404
        
        # EN
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # ENYAMLEN
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.warning(f"YAMLEN: {e}")
                config_data = {}
        
        return jsonify({
            "success": True,
            "config_name": config_name,
            "config_path": config_path,
            "config_content": config_content,
            "config_data": config_data,
            "message": f"EN: {config_name}.yaml"
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/start-training-with-config-id/<config_id>', methods=['POST'])
def start_training_with_config_id(config_id: str):
    """ENIDEN(ENgenerated_configsEN)"""
    try:
        # EN
        data = request.json if request.json else {}
        auto_validation = data.get('auto_validation', False)  # EN
        
        print(f"\n [API] EN(ENID)")
        print(f" ENID: {config_id}")
        print(f" EN: {auto_validation}")
        
        # ENIDEN
        if config_id not in training_configs:
            error_msg = f"ENID {config_id} EN"
            print(f" {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        config_info = training_configs[config_id]
        config_path = config_info['config_path']
        model_name = config_info['model_name']
        
        print(f" EN: {config_path}")
        print(f" EN: {model_name}")
        
        # EN
        if not os.path.exists(config_path):
            error_msg = f"EN {config_path} EN"
            print(f" {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        # ENID
        job_id = str(uuid.uuid4())
        print(f" ENID: {job_id}")
        
        # EN
        job = TrainingJob(
            job_id=job_id,
            config_path=config_path,
            model_name=model_name
        )
        
        # EN
        job.auto_validation = auto_validation
        
        # EN
        if 'config' in config_info and config_info['config']:
            config_data = config_info['config']
            if 'TRAIN' in config_data:
                job.total_epochs = config_data['TRAIN'].get('EPOCHS', 10)
            else:
                job.total_epochs = 10
        else:
            job.total_epochs = 10
        
        print(f" EN: {job.total_epochs}")
        
        training_jobs[job_id] = job
        
        print(f" EN...")
        
        # EN
        training_thread = threading.Thread(
            target=run_training,
            args=(job,),
            daemon=True
        )
        training_thread.start()
        
        validation_message = ",EN" if auto_validation else ""
        success_msg = f"EN: {model_name} (ENID {config_id}){validation_message}"
        print(f" {success_msg}")
        
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
        error_msg = f"ENIDEN: {e}"
        logger.error(error_msg)
        print(f" {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/start-training-with-config/<config_name>', methods=['POST'])
def start_training_with_existing_config(config_name: str):
    """EN"""
    try:
        # EN
        data = request.json if request.json else {}
        auto_validation = data.get('auto_validation', False)  # EN
        
        print(f"\n [API] EN")
        print(f" EN: {config_name}.yaml")
        print(f" EN: {auto_validation}")
        
        # EN
        config_path = os.path.join('configs', f'{config_name}.yaml')
        
        if not os.path.exists(config_path):
            error_msg = f"EN {config_name}.yaml EN"
            print(f" {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        # EN
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            model_name = config_data.get('MODEL', {}).get('NAME', config_name)
        except:
            model_name = config_name
        
        print(f" EN: {model_name}")
        
        # ENID
        job_id = str(uuid.uuid4())
        print(f" ENID: {job_id}")
        
        # EN
        job = TrainingJob(
            job_id=job_id,
            config_path=config_path,
            model_name=model_name
        )
        
        # EN
        job.auto_validation = auto_validation
        
        # EN
        if config_data and 'TRAIN' in config_data:
            job.total_epochs = config_data['TRAIN'].get('EPOCHS', 10)
        else:
            job.total_epochs = 10
        
        print(f" EN: {job.total_epochs}")
        
        training_jobs[job_id] = job
        
        print(f" EN...")
        
        # EN
        training_thread = threading.Thread(
            target=run_training,
            args=(job,),
            daemon=True
        )
        training_thread.start()
        
        validation_message = ",EN" if auto_validation else ""
        success_msg = f"EN: {model_name} (EN {config_name}.yaml){validation_message}"
        print(f" {success_msg}")
        
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
        error_msg = f"EN: {e}"
        logger.error(error_msg)
        print(f" {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/list-configs', methods=['GET'])
def list_existing_configs():
    """EN"""
    try:
        config_dir = Path('configs')
        if not config_dir.exists():
            return jsonify({
                "success": True,
                "configs": [],
                "message": "EN"
            })
        
        # ENyamlEN
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
                logger.warning(f"EN {config_file} EN: {e}")
                continue
        
        return jsonify({
            "success": True,
            "configs": config_files,
            "message": f"EN {len(config_files)} EN"
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def run_validation(job: TrainingJob):
    """EN"""
    try:
        job.status = "running"
        job.start_time = datetime.now().isoformat()
        job.status_description = "EN..."
        job.add_log(f" EN: {job.model_name}")
        
        # EN
        print(f"\n{'='*80}")
        print(f" vulcan EN")
        print(f"{'='*80}")
        print(f" EN:")
        print(f"   • ENID: {job.job_id}")
        print(f"   • EN: {job.model_name}")
        print(f"   • EN: {job.config_path}")
        print(f"   • EN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # EN
        current_dir = os.path.dirname(__file__)
        project_root = current_dir
        
        job.add_log(f" EN: {project_root}")
        print(f" EN: {project_root}")
        
        # EN - EN(EN tools/ EN)
        val_script = None
        config_path = os.path.abspath(job.config_path)
        
        job.add_log(" EN python -m vulcan.cli.val EN")
        print(" EN python -m vulcan.cli.val EN")
        
        job.add_log(f" EN: {config_path}")
        print(f" EN: {config_path}")
        if not os.path.exists(config_path):
            error_msg = f" EN: {config_path}"
            job.add_log(error_msg)
            print(error_msg)
            job.status = "failed"
            job.status_description = "EN"
            return
            
        job.add_log(f" EN")
        print(f" EN")
        
        # EN
        output_dir = os.path.join(project_root, "output")
        job.add_log(f" EN: {output_dir}")
        print(f" EN: {output_dir}")
        if os.path.exists(output_dir):
            model_files = list(Path(output_dir).glob("*.pth")) + list(Path(output_dir).glob("*.pt"))
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                job.add_log(f" EN: {latest_model}")
                print(f" EN: {latest_model}")
            else:
                job.add_log(f" EN (.pth/.pt)")
                print(f" EN (.pth/.pt)")
        else:
            job.add_log(f" EN: {output_dir}")
            print(f" EN: {output_dir}")
        
        # EN
        cmd = [
            sys.executable,
            val_script,
            "--cfg", config_path
        ]
        
        job.add_log(f" EN: {' '.join(cmd)}")
        print(f" EN: {' '.join(cmd)}")
        job.status_description = "EN..."
        
        # EN
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # EN
        env['PYTHONIOENCODING'] = 'utf-8'  # EN
        
        # EN
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
        
        print(f" EN,PID: {process.pid}")
        job.add_log(f" EN,PID: {process.pid}")
        print(f"\n EN:")
        print(f"{'='*60}")
        
        # EN
        try:
            for line in process.stdout:
                line = line.strip()
                if line:
                    # EN
                    print(f" {line}")
                    job.add_log(line)
                    
                    # EN - EN "39% | 886/2273" EN
                    if '%' in line and ('|' in line or '/' in line):
                        try:
                            # EN
                            percent_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                            if percent_match:
                                progress_percent = float(percent_match.group(1))
                                job.progress = progress_percent
                                print(f" EN: {progress_percent:.1f}%")
                                job.add_log(f"EN: {progress_percent:.1f}%")
                            
                            # EN (EN: 886/2273)
                            step_match = re.search(r'(\d+)/(\d+)', line)
                            if step_match:
                                current_step = int(step_match.group(1))
                                total_steps = int(step_match.group(2))
                                job.current_iteration = current_step
                                job.total_iterations = total_steps
                                if total_steps > 0:
                                    step_progress = (current_step / total_steps) * 100
                                    # EN,EN
                                    if job.progress == 0:
                                        job.progress = step_progress
                                        print(f" EN: {current_step}/{total_steps} ({step_progress:.1f}%)")
                                        job.add_log(f"EN: {current_step}/{total_steps} ({step_progress:.1f}%)")
                                    else:
                                        print(f" EN: {current_step}/{total_steps}")
                                        job.add_log(f"EN: {current_step}/{total_steps}")
                        except Exception as e:
                            job.add_log(f"EN: {e}")
                    
                    # EN - EN
                    if "Accuracy:" in line or "accuracy:" in line or "acc:" in line:
                        try:
                            # EN: Accuracy: 0.85, accuracy: 85%, acc: 0.8500
                            patterns = [r'Accuracy[:\s]+([\d\.]+)', r'accuracy[:\s]+([\d\.]+)', r'acc[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                acc_match = re.search(pattern, line, re.IGNORECASE)
                                if acc_match:
                                    acc_value = float(acc_match.group(1))
                                    # EN1,EN
                                    if acc_value > 1:
                                        acc_value = acc_value / 100
                                    job.metrics["validation_accuracy"] = acc_value
                                    print(f" EN: {acc_value:.4f}")
                                    job.add_log(f"EN: {acc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENaccuracyEN: {e} - EN: {line}")
                    
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
                                    print(f" ENF1: {f1_value:.4f}")
                                    job.add_log(f"ENF1: {f1_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENF1EN: {e} - EN: {line}")
                    
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
                                    print(f" EN: {prec_value:.4f}")
                                    job.add_log(f"EN: {prec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENprecisionEN: {e} - EN: {line}")
                    
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
                                    print(f" EN: {rec_value:.4f}")
                                    job.add_log(f"EN: {rec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENrecallEN: {e} - EN: {line}")
                    
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
                                    print(f" ENAUC: {auc_value:.4f}")
                                    job.add_log(f"ENAUC: {auc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENAUCEN: {e} - EN: {line}")
                    
                    # EN
                    if any(error_word in line.lower() for error_word in ['error', 'exception', 'failed', 'traceback']):
                        print(f" EN: {line}")
                        job.add_log(f"EN: {line}")
        
        except Exception as e:
            error_msg = f"EN: {e}"
            job.add_log(error_msg)
            print(f" {error_msg}")
        
        print(f"{'='*60}")
        print(f" EN...")
        
        # EN
        return_code = process.wait()
        
        print(f" EN,EN: {return_code}")
        
        # EN - EN
        has_metrics = any(key.startswith('validation_') for key in job.metrics.keys())
        
        if return_code == 0 or has_metrics:
            job.status = "completed"
            job.add_log("EN!")
            print(f" EN!")
            
            # EN100%
            job.progress = 100
            
            # EN
            print(f"\n EN:")
            job.add_log("EN:")
            if job.metrics.get("validation_accuracy", 0) > 0:
                acc_msg = f"• EN: {job.metrics['validation_accuracy']:.4f}"
                job.add_log(acc_msg)
                print(f"  {acc_msg}")
            if job.metrics.get("validation_f1", 0) > 0:
                f1_msg = f"• ENF1: {job.metrics['validation_f1']:.4f}"
                job.add_log(f1_msg)
                print(f"  {f1_msg}")
            if job.metrics.get("validation_precision", 0) > 0:
                prec_msg = f"• EN: {job.metrics['validation_precision']:.4f}"
                job.add_log(prec_msg)
                print(f"  {prec_msg}")
            if job.metrics.get("validation_recall", 0) > 0:
                rec_msg = f"• EN: {job.metrics['validation_recall']:.4f}"
                job.add_log(rec_msg)
                print(f"  {rec_msg}")
            if job.metrics.get("validation_auc", 0) > 0:
                auc_msg = f"• ENAUC: {job.metrics['validation_auc']:.4f}"
                job.add_log(auc_msg)
                print(f"  {auc_msg}")
                
            # EN,EN
            if not has_metrics:
                warning_msg = "EN: EN,EN"
                job.add_log(warning_msg)
                print(f" {warning_msg}")
        else:
            job.status = "failed"
            failure_msg = f"EN,EN: {return_code}"
            job.add_log(failure_msg)
            print(f" {failure_msg}")
            # EN
            job.add_log("EN,EN:")
            job.add_log("1. EN")
            job.add_log("2. EN")
            job.add_log("3. EN")
            job.add_log("4. EN")
            print(f" EN:")
            print(f"  1. EN")
            print(f"  2. EN")
            print(f"  3. EN")
            print(f"  4. EN")
        
        job.end_time = datetime.now().isoformat()
        duration_msg = f" EN: {job.get_duration()}"
        job.add_log(duration_msg)
        print(f"{duration_msg}")
        
        print(f"{'='*80}")
        print(f" vulcan EN")
        print(f"{'='*80}\n")
        
        # EN
        job.save_logs_to_file()
        
    except Exception as e:
        job.status = "failed"
        job.status_description = f"EN: {str(e)}"
        job.add_log(f" EN: {str(e)}")
        job.end_time = datetime.now().isoformat()
        logger.error(f"EN {job.job_id} EN: {e}", exc_info=True)
        print(f" [EN] {str(e)}")
        
        # EN
        import traceback
        job.add_log(" EN:")
        job.add_log(traceback.format_exc())
        
        print(f" EN:")
        print(traceback.format_exc())
        
        # EN
        job.save_logs_to_file()

@app.route('/api/health', methods=['GET'])
def health_check():
    """EN"""
    return jsonify({
        "success": True,
        "message": "vulcan Backend API Server is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/start-validation/<config_name>', methods=['POST'])
def start_validation_with_config(config_name: str):
    """EN"""
    try:
        print(f"\n [API] EN")
        print(f" EN: {config_name}.yaml")
        
        # EN
        if not ensure_validation_script_format():
            logger.warning("EN,EN")
            print(f" EN,EN")
        
        # EN
        config_path = os.path.join('configs', f'{config_name}.yaml')
        
        if not os.path.exists(config_path):
            error_msg = f"EN {config_name}.yaml EN"
            print(f" {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        # EN
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            model_name = config_data.get('MODEL', {}).get('NAME', config_name)
        except:
            model_name = config_name
        
        print(f" EN: {model_name}")
        
        # ENID
        job_id = str(uuid.uuid4())
        print(f" ENID: {job_id}")
        
        # EN
        job = TrainingJob(
            job_id=job_id,
            config_path=config_path,
            model_name=model_name
        )
        
        # EN
        job.is_validation_task = True
        job.total_epochs = 1  # EN1ENepoch
        job.progress = 0  # EN0
        
        training_jobs[job_id] = job
        
        print(f" EN...")
        
        # EN
        validation_thread = threading.Thread(
            target=run_validation,
            args=(job,),
            daemon=True
        )
        validation_thread.start()
        
        success_msg = f"EN: {model_name} (EN {config_name}.yaml)"
        print(f" {success_msg}")
        
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
        error_msg = f"EN: {e}"
        logger.error(error_msg)
        print(f" {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/start-validation-with-config-id/<config_id>', methods=['POST'])
def start_validation_with_config_id(config_id: str):
    """ENIDEN(ENgenerated_configsEN)"""
    try:
        print(f"\n [API] EN(ENID)")
        print(f" ENID: {config_id}")
        
        # EN
        if not ensure_validation_script_format():
            logger.warning("EN,EN")
            print(f" EN,EN")
        
        # ENIDEN
        if config_id not in training_configs:
            error_msg = f"ENID {config_id} EN"
            print(f" {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        config_info = training_configs[config_id]
        config_path = config_info['config_path']
        model_name = config_info['model_name']
        
        print(f" EN: {config_path}")
        print(f" EN: {model_name}")
        
        # EN
        if not os.path.exists(config_path):
            error_msg = f"EN {config_path} EN"
            print(f" {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        # ENID
        job_id = str(uuid.uuid4())
        print(f" ENID: {job_id}")
        
        # EN
        job = TrainingJob(
            job_id=job_id,
            config_path=config_path,
            model_name=model_name
        )
        
        # EN
        job.is_validation_task = True
        job.total_epochs = 1  # EN1ENepoch
        job.progress = 0  # EN0
        
        training_jobs[job_id] = job
        
        print(f" EN...")
        
        # EN
        validation_thread = threading.Thread(
            target=run_validation,
            args=(job,),
            daemon=True
        )
        validation_thread.start()
        
        success_msg = f"EN: {model_name} (ENID {config_id})"
        print(f" {success_msg}")
        
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
        error_msg = f"ENIDEN: {e}"
        logger.error(error_msg)
        print(f" {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/validation-status/<job_id>', methods=['GET'])
def get_validation_status(job_id: str):
    """EN"""
    try:
        print(f" [API] EN: {job_id}")
        
        if job_id not in training_jobs:
            error_msg = "EN"
            print(f" {error_msg}: {job_id}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 404
        
        job = training_jobs[job_id]
        
        # EN - EN
        is_validation_task = (
            hasattr(job, 'is_validation_task') and job.is_validation_task
        ) or (
            # EN,EN
            any(key.startswith('validation_') for key in job.metrics.keys())
        )
        
        print(f" EN: {job.status}, EN: {job.progress}%, EN: {is_validation_task}")
        
        # EN
        validation_metrics = {}
        for key, value in job.metrics.items():
            if key.startswith('validation_'):
                validation_metrics[key] = value
        
        if validation_metrics:
            print(f" EN: {validation_metrics}")
        
        # EN
        logs = job.get_recent_logs(100)
        
        # EN
        response_data = {
            "success": True,
            "job_id": job_id,
            "task_type": "validation",
            "status": job.status,
            "progress": job.progress,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "current_iteration": job.current_iteration,  # EN
            "total_iterations": job.total_iterations,    # EN
            "start_time": job.start_time,
            "end_time": job.end_time,
            "logs": logs,
            "log_count": len(logs),
            "log_file_path": job.get_log_file_path(),
            
            # EN
            "validation_metrics": validation_metrics,
            
            # EN
            "model_name": job.model_name,
            "config_path": job.config_path,
            
            # EN
            "status_description": get_validation_status_description(job.status),
            
            # EN
            "validation_summary": generate_validation_summary(validation_metrics)
        }
        
        # EN,EN
        if job.status == "completed":
            response_data["final_validation_results"] = generate_final_validation_results(validation_metrics, job)
            print(f" EN,EN")
        
        # EN,EN
        if job.status == "failed":
            error_logs = [log for log in logs if any(keyword in log.lower() for keyword in ['error', 'exception', 'failed', 'traceback', 'EN', 'EN', 'EN'])]
            response_data["error_details"] = {
                "error_logs": error_logs[-10:] if error_logs else [],  # EN10EN
                "troubleshooting_tips": [
                    "EN output EN",
                    "EN",
                    "EN",
                    "EN",
                    "EN"
                ]
            }
            print(f" EN,EN: {len(error_logs)}")
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"EN: {e}"
        logger.error(error_msg)
        print(f" {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def get_validation_status_description(status: str) -> str:
    """EN"""
    if status == "pending":
        return "EN"
    elif status == "running":
        return "EN"
    elif status == "completed":
        return "EN"
    elif status == "failed":
        return "EN"
    else:
        return "EN"

def generate_validation_summary(validation_metrics: dict) -> dict:
    """EN"""
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
    """EN"""
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
    
    # EN - EN(EN)
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
    
    # EN
    if validation_metrics:
        acc = validation_metrics.get("validation_accuracy", 0)
        f1 = validation_metrics.get("validation_f1", 0)
        
        if acc > 0.9:
            final_results["performance_assessment"].append(" EN!EN90%")
        elif acc > 0.8:
            final_results["performance_assessment"].append(" EN,EN80%")
        elif acc > 0.7:
            final_results["performance_assessment"].append(" EN,EN70-80%EN")
        else:
            final_results["performance_assessment"].append(" EN,EN70%")
        
        if f1 > 0 and abs(acc - f1) > 0.1:
            final_results["performance_assessment"].append(" ENF1EN,EN")
    
    # EN
    if validation_metrics:
        acc = validation_metrics.get("validation_accuracy", 0)
        f1 = validation_metrics.get("validation_f1", 0)
        
        if acc > 0.9:
            final_results["recommendations"].append(" EN,EN")
        elif acc > 0.8:
            final_results["recommendations"].append(" EN,EN")
        elif acc > 0.7:
            final_results["recommendations"].append(" EN")
        else:
            final_results["recommendations"].append(" EN")
        
        if f1 > 0 and f1 < 0.6:
            final_results["recommendations"].append(" EN,EN")
    
    return final_results

class OptimizationJob:
    """EN"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"  # pending, running, completed, failed
        self.status_description = "EN..."
        self.progress = 0
        self.current_iteration = 0
        self.total_iterations = 15
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.process = None
        self.best_ratio = 0.0
        self.best_f1_score = 0.0
        self.current_ratio = 0.0  # EN
        self.current_f1 = 0.0    # ENF1EN
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
            logger.error(f"EN: {e}")
        
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
            logger.error(f"EN: {e}")
        
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
        job.status_description = "EN..."
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
                    print(f"     {dataset_file}: EN")
                    job.add_log(f" EN: {dataset_file}")
        else:
            print(f"    EN: {dataset_dir}")
            job.add_log(f" EN: {dataset_dir}")
        
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
                print(f"     GPU: EN,ENCPU")
                job.add_log(f" GPUEN,ENCPU")
        except ImportError:
            print(f"     PyTorchEN,ENGPU")
            job.add_log(f" PyTorchEN,ENGPU")
        
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
            cwd=project_root,
            env=dict(os.environ, PYTHONUNBUFFERED='1')  # ENPythonEN
        )
        
        job.process = process
        
        print(f" EN,PID: {process.pid}")
        job.add_log(f" EN,PID: {process.pid}")
        
        # EN
        patterns = {
            'iteration': r'EN(\d+)EN',
            'ratio': r'EN\s+([0-9.]+)',
            'f1_score': r'F1EN:\s+([0-9.]+)',
            'best_ratio': r'EN:\s+([0-9.]+)',
            'best_f1': r'F1EN:\s+([0-9.]+)',
            'progress': r'EN[::]\s*(\d+)%',
            'completed': r'EN',
            'failed': r'EN|EN|Error|Exception'
        }
        
        current_ratio = 0.0
        last_output_time = time.time()
        no_output_timeout = 60  # 60EN
        
        # EN
        while True:
            # EN
            if process.poll() is not None:
                print(f" EN,EN: {process.returncode}")
                job.add_log(f" EN,EN: {process.returncode}")
                break
            
            # EN
            if time.time() - last_output_time > no_output_timeout:
                print(f" EN: {no_output_timeout}EN,EN")
                job.add_log(f" EN: {no_output_timeout}EN,EN")
                # EN,EN
            
            # EN
            try:
                line = process.stdout.readline()
                if line:
                    last_output_time = time.time()
                    # EN
                    print(line.rstrip())
                    job.add_log(line.rstrip())
                    
                    # EN
                    for key, pattern in patterns.items():
                        match = re.search(pattern, line)
                        if match:
                            if key == 'iteration':
                                job.current_iteration = int(match.group(1))
                                job.progress = min(100, int((job.current_iteration / job.total_iterations) * 100))
                                print(f" EN: {job.progress}% (EN{job.current_iteration}EN)")
                            elif key == 'ratio':
                                job.current_ratio = float(match.group(1))
                                print(f" EN: {job.current_ratio:.3f}")
                            elif key == 'f1_score':
                                job.current_f1 = float(match.group(1))
                                print(f" ENF1EN: {job.current_f1:.4f}")
                                if job.current_f1 > job.best_f1_score:
                                    job.best_f1_score = job.current_f1
                                    job.best_ratio = job.current_ratio
                                    print(f" EN: EN={job.best_ratio:.3f}, F1={job.best_f1_score:.4f}")
                            elif key == 'best_ratio':
                                job.best_ratio = float(match.group(1))
                            elif key == 'completed':
                                job.status = "completed"
                                job.status_description = "EN"
                                job.progress = 100
                                print(" EN")
                            elif key == 'failed':
                                job.status = "failed"
                                job.status_description = "EN"
                                print(" EN")
                else:
                    # EN,EN
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f" EN: {e}")
                job.add_log(f" EN: {e}")
                break
        
        # EN
        try:
            return_code = process.wait(timeout=30)  # 30EN
        except subprocess.TimeoutExpired:
            print(" EN,EN")
            job.add_log(" EN,EN")
            process.kill()
            return_code = -1
        
        job.end_time = datetime.now().isoformat()
        
        print(f" EN,EN: {return_code}")
        job.add_log(f" EN,EN: {return_code}")
        
        if return_code == 0 and job.status != "failed":
            job.status = "completed"
            job.status_description = f"EN,EN: {job.best_ratio:.3f}, F1: {job.best_f1_score:.4f}"
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
            job.status_description = f"EN,EN: {return_code}"
            error_msg = f"\n EN!EN: {return_code}"
            print(error_msg)
            job.add_log(error_msg)
            
    except Exception as e:
        job.status = "failed"
        job.status_description = f"EN: {str(e)}"
        job.end_time = datetime.now().isoformat()
        error_msg = f"EN: {str(e)}"
        print(error_msg)
        job.add_log(error_msg)
        logger.error(f"EN: {e}")
        import traceback
        job.add_log(traceback.format_exc())

@app.route('/api/start-dataset-optimization', methods=['POST'])
def start_dataset_optimization():
    """EN"""
    try:
        data = request.json
        max_iterations = data.get('max_iterations', 15)
        
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
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "EN"
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/optimization-status/<job_id>', methods=['GET'])
def get_optimization_status(job_id: str):
    """EN"""
    try:
        if job_id not in optimization_jobs:
            return jsonify({
                "success": False,
                "error": "EN"
            }), 404
        
        job = optimization_jobs[job_id]
        
        # EN
        if job.status == "completed":
            logs = job.get_full_logs()
            log_count = len(logs)
        elif job.status == "failed":
            logs = job.get_full_logs()
            log_count = len(logs)
        else:
            # EN
            logs = job.get_logs(limit=100)  # EN100EN
            log_count = len(logs)
        
        # EN
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
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/optimization-logs/<job_id>', methods=['GET'])
def get_optimization_logs(job_id: str):
    """EN"""
    try:
        if job_id not in optimization_jobs:
            return jsonify({
                "success": False,
                "error": "EN"
            }), 404
        
        job = optimization_jobs[job_id]
        logs = job.get_full_logs()  # EN
        
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
        logger.error(f"EN: {e}")
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
        self.status_description = 'EN...'
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = 0
        self.progress = 0  # EN

    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

def run_paper_search(job: PaperSearchJob):
    try:
        job.status = 'running'
        job.status_description = 'EN...'
        job.start_time = datetime.now()
        job.add_log("EN...")
        job.progress = 0
        
        # ENResourceUpdaterEN
        from resource_updater import ResourceUpdater
        
        # ENResourceUpdaterEN
        updater = ResourceUpdater()
        
        job.add_log(f"EN: {job.search_query}")
        job.add_log(f"EN: {job.year_from}")
        job.add_log(f"EN: {job.top_k}")
        job.progress = 20
        
        # EN
        job.add_log("EN...")
        job.progress = 40
        
        papers = updater.semantic_search_papers(
            query=job.search_query,
            year_from=job.year_from,
            top_k=job.top_k
        )
        
        job.progress = 80
        job.add_log(f"EN,EN {len(papers)} EN")
        
        # EN
        for i, paper in enumerate(papers, 1):
            job.add_log(f"EN {i}: {paper.get('title', 'Unknown Title')}")
            job.papers.append(paper)
        
        job.progress = 100
        job.status = 'completed'
        job.status_description = f'EN,EN {len(papers)} EN'
        job.end_time = datetime.now()
        job.duration = (job.end_time - job.start_time).total_seconds()
        job.add_log("EN!")
        
    except Exception as e:
        job.status = 'failed'
        job.status_description = f'EN: {str(e)}'
        job.end_time = datetime.now()
        job.duration = (job.end_time - job.start_time).total_seconds()
        job.progress = 0
        job.add_log(f"EN: {e}")
        import traceback
        job.add_log(f"EN: {traceback.format_exc()}")

@app.route('/api/start-paper-search', methods=['POST'])
def start_paper_search():
    try:
        data = request.json
        search_query = data.get('search_query', 'deep learning vulnerability detection')
        year_from = data.get('year_from', 2023)
        top_k = data.get('top_k', 5)
        
        job_id = str(uuid.uuid4())
        job = PaperSearchJob(job_id, search_query, year_from, top_k)
        
        # ENjobENjobsEN
        jobs[job_id] = job
        
        # EN
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
        logger.error(f"EN: {e}")
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
                "error": "EN"
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
        logger.error(f"EN: {e}")
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
                "error": "EN"
            }), 404
        
        job = jobs[job_id]
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "logs": job.logs
        })
    except Exception as e:
        logger.error(f"EN: {e}")
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
                "error": "EN"
            }), 404
        
        job = jobs[job_id]
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "papers": job.papers
        })
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ENAPIEN

class DataCollectionJob:
    """EN"""
    def __init__(self, job_id: str, collection_type: str = "comprehensive"):
        self.job_id = job_id
        self.collection_type = collection_type
        self.status = "pending"  # pending, running, completed, failed
        self.status_description = "EN"
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.collection_results = {}
        self.error = None
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """EN"""
        import os
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def add_log(self, message: str):
        """EN"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # EN
        if len(self.logs) > 1000:
            self.logs = self.logs[-500:]
    
    def get_logs(self, limit: int = None) -> List[str]:
        """EN"""
        if limit:
            return self.logs[-limit:]
        return self.logs
    
    def get_recent_logs(self, count: int = 50) -> List[str]:
        """EN"""
        return self.logs[-count:]
    
    def get_duration(self) -> str:
        """EN"""
        if not self.start_time:
            return "EN"
        if not self.end_time:
            end_time = datetime.now()
        else:
            end_time = self.end_time
        
        duration = end_time - self.start_time
        return str(duration).split('.')[0]  # EN

def run_data_collection(job: DataCollectionJob):
    """EN"""
    try:
        job.status = "running"
        job.status_description = "EN"
        job.start_time = datetime.now()
        job.add_log(" EN")
        
        # EN
        from data_collector import DataCollector
        
        job.add_log(" EN...")
        collector = DataCollector(job.collection_type)
        
        job.add_log(" EN...")
        
        # EN
        result = collector.collect_sample_data()
        
        if result["success"]:
            job.status = "completed"
            job.status_description = "EN"
            job.collection_results = result
            job.add_log(" EN")
            job.add_log(f" EN: {result['stats']}")
            job.add_log(f" EN: {result['output_file']}")
        else:
            job.status = "failed"
            job.status_description = "EN"
            job.error = "EN"
            job.add_log(" EN")
            
    except Exception as e:
        job.status = "failed"
        job.status_description = f"EN: {str(e)}"
        job.error = str(e)
        job.add_log(f" EN: {str(e)}")
        logger.error(f"EN: {e}")
        traceback.print_exc()
    
    finally:
        job.end_time = datetime.now()
        if job.start_time:
            job.duration = (job.end_time - job.start_time).total_seconds()

@app.route('/api/start-data-collection', methods=['POST'])
def start_data_collection():
    """EN"""
    try:
        data = request.get_json()
        collection_type = data.get('collection_type', 'comprehensive')
        
        # ENID
        job_id = str(uuid.uuid4())
        
        # EN
        job = DataCollectionJob(job_id, collection_type)
        jobs[job_id] = job
        
        # EN
        thread = threading.Thread(target=run_data_collection, args=(job,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"EN: {job_id}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": f"EN,ENID: {job_id}",
            "collection_type": collection_type
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-collection-status/<job_id>', methods=['GET'])
def get_data_collection_status(job_id: str):
    """EN"""
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "EN"
            }), 404
        
        job = jobs[job_id]
        
        # EN
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
        
        # EN,EN
        if job.status == "completed" and job.collection_results:
            response_data["collection_results"] = job.collection_results
        
        # EN,EN
        if job.status == "failed" and job.error:
            response_data["error"] = job.error
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-collection-logs/<job_id>', methods=['GET'])
def get_data_collection_logs(job_id: str):
    """EN"""
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "EN"
            }), 404
        
        job = jobs[job_id]
        
        # EN
        limit = request.args.get('limit', type=int)
        
        logs = job.get_logs(limit)
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "logs": logs,
            "total_logs": len(job.logs)
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-collection-results/<job_id>', methods=['GET'])
def get_data_collection_results(job_id: str):
    """EN"""
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "EN"
            }), 404
        
        job = jobs[job_id]
        
        if job.status != "completed":
            return jsonify({
                "success": False,
                "error": "EN"
            }), 400
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "results": job.collection_results
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def ensure_validation_script_format():
    """EN"""
    try:
        val_script_path = os.path.join(os.path.dirname(__file__), "tools", "val.py")
        if not os.path.exists(val_script_path):
            logger.warning(f"EN: {val_script_path}")
            return False
        
        # EN
        with open(val_script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # EN
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
            logger.warning(f"EN: {missing_patterns}")
            return False
        
        logger.info("EN")
        return True
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return False

@app.route('/api/test-validation', methods=['GET'])
def test_validation_setup():
    """EN"""
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
        
        # EN
        val_script_path = os.path.join(os.path.dirname(__file__), "tools", "val.py")
        test_results["validation_script_exists"] = os.path.exists(val_script_path)
        if not test_results["validation_script_exists"]:
            test_results["issues"].append("EN(EN console script: vulcan-val)")
            test_results["recommendations"].append("EN: vulcan-val --cfg <path>")
        else:
            test_results["validation_script_format_ok"] = ensure_validation_script_format()
            if not test_results["validation_script_format_ok"]:
                test_results["issues"].append("EN")
                test_results["recommendations"].append("EN")
        
        # EN
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        test_results["output_directory_exists"] = os.path.exists(output_dir)
        if test_results["output_directory_exists"]:
            # EN
            model_files = list(Path(output_dir).glob("*.pth")) + list(Path(output_dir).glob("*.pt"))
            test_results["model_files_found"] = [str(f) for f in model_files]
            if not model_files:
                test_results["issues"].append("output EN")
                test_results["recommendations"].append("EN")
        else:
            test_results["issues"].append("output EN")
            test_results["recommendations"].append("EN output EN")
        
        # EN
        configs_dir = os.path.join(os.path.dirname(__file__), "configs")
        if os.path.exists(configs_dir):
            config_files = list(Path(configs_dir).glob("*.yaml"))
            test_results["config_files_found"] = [f.stem for f in config_files]
            if not config_files:
                test_results["issues"].append("configs EN")
                test_results["recommendations"].append("EN configs EN")
        else:
            test_results["issues"].append("configs EN")
            test_results["recommendations"].append("EN configs EN")
        
        # EN
        dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
        if os.path.exists(dataset_dir):
            dataset_files = ["train.jsonl", "valid.jsonl", "test.jsonl"]
            missing_datasets = []
            for dataset_file in dataset_files:
                if not os.path.exists(os.path.join(dataset_dir, dataset_file)):
                    missing_datasets.append(dataset_file)
            
            if missing_datasets:
                test_results["issues"].append(f"EN: {', '.join(missing_datasets)}")
                test_results["recommendations"].append("EN")
        else:
            test_results["issues"].append("dataset EN")
            test_results["recommendations"].append("EN dataset EN")
        
        # EN
        test_results["overall_status"] = "ready" if not test_results["issues"] else "needs_attention"
        
        return jsonify({
            "success": True,
            "test_results": test_results,
            "message": "EN" if not test_results["issues"] else f"EN {len(test_results['issues'])} EN"
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "test_results": None
        }), 500

def run_training(job: TrainingJob):
    """EN(EN,EN)"""
    try:
        job.status = "running"
        job.start_time = datetime.now().isoformat()
        job.status_description = "EN..."
        job.add_log(f" EN: {job.model_name}")
        
        # EN
        print(f"\n{'='*80}")
        print(f" vulcan EN")
        print(f"{'='*80}")
        print(f" EN:")
        print(f"   • ENID: {job.job_id}")
        print(f"   • EN: {job.model_name}")
        print(f"   • EN: {job.config_path}")
        print(f"   • EN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # EN
        current_dir = os.path.dirname(__file__)
        project_root = current_dir
        
        job.add_log(f" EN: {project_root}")
        print(f" EN: {project_root}")
        
        # EN - EN
        train_script = None
        config_path = os.path.abspath(job.config_path)
        
        job.add_log(" EN: python -m vulcan.cli.train")
        print(" EN: python -m vulcan.cli.train")
        
        job.add_log(f" EN: {config_path}")
        print(f" EN: {config_path}")
        if not os.path.exists(config_path):
            error_msg = f" EN: {config_path}"
            job.add_log(error_msg)
            print(error_msg)
            job.status = "failed"
            job.status_description = "EN"
            return
            
        job.add_log(f" EN")
        print(f" EN")
        
        # EN
        dataset_dir = os.path.join(project_root, "dataset")
        job.add_log(f" EN: {dataset_dir}")
        print(f" EN: {dataset_dir}")
        if os.path.exists(dataset_dir):
            job.add_log(f" EN")
            print(f" EN")
        else:
            job.add_log(f" EN: {dataset_dir}")
            print(f" EN: {dataset_dir}")
        
        # EN
        cmd = [sys.executable, "-m", "vulcan.cli.train", "--cfg", config_path]
        
        job.add_log(f" EN: {' '.join(cmd)}")
        print(f" EN: {' '.join(cmd)}")
        job.status_description = "EN..."
        
        # EN
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # EN
        env['PYTHONIOENCODING'] = 'utf-8'  # EN
        
        # EN
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
        
        print(f" EN,PID: {process.pid}")
        job.add_log(f" EN,PID: {process.pid}")
        print(f"\n EN:")
        print(f"{'='*60}")
        
        # EN
        try:
            for line in process.stdout:
                line = line.strip()
                if line:
                    # EN
                    print(f" {line}")
                    job.add_log(line)
                    
                    # EN
                    if 'epoch' in line.lower() or 'Epoch' in line:
                        try:
                            # ENEpochEN: Epoch: [1/1], Epoch 1/10, epoch: 1
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
                                        print(f" EN(Epoch): {current_epoch}/{job.total_epochs} ({job.progress:.1f}%)")
                                        job.add_log(f"EN(Epoch): {current_epoch}/{job.total_epochs} ({job.progress:.1f}%)")
                                    break
                        except Exception as e:
                            job.add_log(f"ENepochEN: {e}")
                    
                    # EN - EN
                    if 'Iter:' in line or 'iter:' in line:
                        try:
                            # EN Iter: [76/142] EN
                            iter_match = re.search(r'[Ii]ter[:\s]*\[(\d+)/(\d+)\]', line)
                            if iter_match:
                                current_iter = int(iter_match.group(1))
                                total_iters = int(iter_match.group(2))
                                job.current_iteration = current_iter
                                job.total_iterations = total_iters
                                
                                # EN
                                if total_iters > 0:
                                    iter_progress = (current_iter / total_iters) * 100
                                    # ENepoch,ENepoch
                                    if job.total_epochs > 1 and job.current_epoch > 0:
                                        # EN = (ENepoch + ENepochEN) / ENepochEN
                                        total_progress = ((job.current_epoch - 1) + (current_iter / total_iters)) / job.total_epochs * 100
                                        job.progress = total_progress
                                    else:
                                        # ENepochEN,EN
                                        job.progress = iter_progress
                                    
                                    print(f" EN: {current_iter}/{total_iters} (EN: {job.progress:.1f}%)")
                                    job.add_log(f"EN: {current_iter}/{total_iters} (EN: {job.progress:.1f}%)")
                        except Exception as e:
                            job.add_log(f"EN: {e}")
                    
                    # EN - EN
                    if '%|' in line:
                        try:
                            # EN: 63%|██████▎
                            percent_match = re.search(r'(\d+)%\|', line)
                            if percent_match:
                                progress_percent = int(percent_match.group(1))
                                # EN
                                if job.progress == 0:
                                    job.progress = progress_percent
                                    print(f" EN: {progress_percent}%")
                                    job.add_log(f"EN: {progress_percent}%")
                        except Exception as e:
                            job.add_log(f"EN: {e}")
                    
                    # EN - EN
                    if "Accuracy:" in line or "accuracy:" in line or "acc:" in line:
                        try:
                            # EN: Accuracy: 0.85, accuracy: 85%, acc: 0.8500
                            patterns = [r'Accuracy[:\s]+([\d\.]+)', r'accuracy[:\s]+([\d\.]+)', r'acc[:\s]+([\d\.]+)']
                            for pattern in patterns:
                                acc_match = re.search(pattern, line, re.IGNORECASE)
                                if acc_match:
                                    acc_value = float(acc_match.group(1))
                                    # EN1,EN
                                    if acc_value > 1:
                                        acc_value = acc_value / 100
                                    job.metrics["validation_accuracy"] = acc_value
                                    print(f" EN: {acc_value:.4f}")
                                    job.add_log(f"EN: {acc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENaccuracyEN: {e} - EN: {line}")
                    
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
                                    print(f" ENF1: {f1_value:.4f}")
                                    job.add_log(f"ENF1: {f1_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENF1EN: {e} - EN: {line}")
                    
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
                                    print(f" EN: {prec_value:.4f}")
                                    job.add_log(f"EN: {prec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENprecisionEN: {e} - EN: {line}")
                    
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
                                    print(f" EN: {rec_value:.4f}")
                                    job.add_log(f"EN: {rec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENrecallEN: {e} - EN: {line}")
                    
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
                                    print(f" ENAUC: {auc_value:.4f}")
                                    job.add_log(f"ENAUC: {auc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENAUCEN: {e} - EN: {line}")
                    
                    # EN
                    if any(error_word in line.lower() for error_word in ['error', 'exception', 'failed', 'traceback']):
                        print(f" EN: {line}")
                        job.add_log(f"EN: {line}")
                    
                    # EN
                    if "loss:" in line.lower() or "Loss:" in line:
                        try:
                            loss_match = re.search(r'[Ll]oss[:\s]+([\d\.]+)', line)
                            if loss_match:
                                loss_value = float(loss_match.group(1))
                                job.metrics["loss"] = loss_value
                                print(f" EN: {loss_value:.4f}")
                                job.add_log(f"EN: {loss_value:.4f}")
                        except Exception as e:
                            job.add_log(f"ENlossEN: {e}")
                    
                    # EN(EN)
                    if ("train" in line.lower() and "acc" in line.lower()) or ("training" in line.lower() and "accuracy" in line.lower()):
                        try:
                            acc_match = re.search(r'[Aa]ccuracy[:\s]+([\d\.]+)', line)
                            if acc_match:
                                acc_value = float(acc_match.group(1))
                                if acc_value > 1:  # EN
                                    acc_value = acc_value / 100
                                job.metrics["accuracy"] = acc_value
                                print(f" EN: {acc_value:.4f}")
                                job.add_log(f"EN: {acc_value:.4f}")
                        except Exception as e:
                            job.add_log(f"ENaccuracyEN: {e}")
        
        except Exception as e:
            error_msg = f"EN: {e}"
            job.add_log(error_msg)
            print(f" {error_msg}")
        
        print(f"{'='*60}")
        print(f" EN...")
        
        # EN
        return_code = process.wait()
        
        print(f" EN,EN: {return_code}")
        
        # EN
        if return_code == 0:
            job.status = "completed"  # EN,EN
            job.add_log(" EN!")
            print(f" EN!")
            
            # EN100%
            job.progress = 100
            
            # EN
            print(f"\n EN:")
            print(f"{'='*60}")
            job.add_log(" EN:")
            job.add_log("="*60)
            
            # EN
            basic_info = f" EN:\n• EN: {job.model_name}\n• EN: {job.config_path}\n• EN: {job.current_epoch}/{job.total_epochs}"
            if job.total_iterations > 0:
                basic_info += f"\n• EN: {job.current_iteration}/{job.total_iterations}"
            basic_info += f"\n• EN: {job.get_duration()}"
            
            print(basic_info)
            job.add_log(basic_info)
            
            # EN
            if job.metrics:
                metrics_info = "\n EN:"
                print(metrics_info)
                job.add_log(metrics_info)
                
                for key, value in job.metrics.items():
                    if not key.startswith('validation_'):  # EN
                        if isinstance(value, (int, float)):
                            metric_line = f"• {key}: {value:.4f}"
                        else:
                            metric_line = f"• {key}: {value}"
                        print(f"  {metric_line}")
                        job.add_log(f"  {metric_line}")
            
            # EN
            output_dir = os.path.join(project_root, "output")
            if os.path.exists(output_dir):
                model_files = list(Path(output_dir).glob("*.pth")) + list(Path(output_dir).glob("*.pt"))
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    model_info = f"\n EN:\n• EN: {output_dir}\n• EN: {latest_model.name}\n• EN: {latest_model.stat().st_size / 1024 / 1024:.2f} MB"
                    print(model_info)
                    job.add_log(model_info)
                    
                    # ====== EN ======
                    if job.auto_validation:
                        print(f"\n EN...")
                        job.add_log("\n EN...")
                        job.status = "running"  # EN,EN
                        job.status_description = "EN..."
                        
                        # EN
                        run_validation_in_training(job, project_root)
                    else:
                        print(f"\n EN,EN")
                        job.add_log("\n EN,EN")
                        job.status = "completed"
                        job.status_description = "EN"
                    
                else:
                    no_model_info = f"\n EN"
                    if job.auto_validation:
                        no_model_info += ",EN"
                    print(no_model_info)
                    job.add_log(no_model_info)
                    job.status = "completed"
            else:
                validation_skip_info = f"\n EN"
                if job.auto_validation:
                    validation_skip_info += ",EN"
                print(validation_skip_info)
                job.add_log(validation_skip_info)
                job.status = "completed"
                
        else:
            job.status = "failed"
            failure_msg = f" EN,EN: {return_code}"
            job.add_log(failure_msg)
            print(failure_msg)
            
            # EN
            print(f"\n EN:")
            print(f"{'='*60}")
            job.add_log(" EN:")
            job.add_log("="*60)
            
            failure_details = f" EN:\n• EN: {job.model_name}\n• EN: {job.config_path}\n• EN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n• EN: {return_code}"
            print(failure_details)
            job.add_log(failure_details)
            
            # EN
            troubleshooting = f"\n EN:\n• 1. EN\n• 2. ENGPU/CUDAEN\n• 3. EN\n• 4. EN\n• 5. EN: {job.get_log_file_path()}\n• 6. ENbatch_sizeEN\n• 7. EN"
            print(troubleshooting)
            job.add_log(troubleshooting)
            
            print(f"{'='*60}")
            job.add_log("="*60)
        
        job.end_time = datetime.now().isoformat()
        duration_msg = f" EN: {job.get_duration()}"
        job.add_log(duration_msg)
        print(f"{duration_msg}")
        
        print(f"{'='*80}")
        print(f" vulcan EN")
        print(f"{'='*80}\n")
        
        # EN
        job.save_logs_to_file()
        
    except Exception as e:
        job.status = "failed"
        job.status_description = f"EN: {str(e)}"
        job.add_log(f" EN: {str(e)}")
        job.end_time = datetime.now().isoformat()
        logger.error(f"EN {job.job_id} EN: {e}", exc_info=True)
        print(f" [EN] {str(e)}")
        
        # EN
        import traceback
        job.add_log(" EN:")
        job.add_log(traceback.format_exc())
        
        print(f" EN:")
        print(traceback.format_exc())
        
        # EN
        job.save_logs_to_file()

def run_validation_in_training(job: TrainingJob, project_root: str):
    """EN,EN"""
    try:
        # EN
        job.current_phase = "validation"
        job.status_description = "EN..."
        
        # EN
        job.add_log("\n" + "="*80)
        job.add_log(" vulcan EN")
        job.add_log("="*80)
        print(f"\n{'='*80}")
        print(f" vulcan EN")
        print(f"{'='*80}")
        
        # EN - EN(EN tools/ EN)
        val_script = None
        config_path = os.path.abspath(job.config_path)
        
        job.add_log(" EN: python -m vulcan.cli.val")
        print(" EN: python -m vulcan.cli.val")
        job.add_log(f" EN: {config_path}")
        print(f" EN: {config_path}")
        
        # EN
        cmd = [sys.executable, "-m", "vulcan.cli.val", "--cfg", config_path]
        
        job.add_log(f" EN: {' '.join(cmd)}")
        print(f" EN: {' '.join(cmd)}")
        
        # EN
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=project_root,
            env=dict(os.environ, PYTHONUNBUFFERED='1')
        )
        
        job.add_log(f" EN,PID: {process.pid}")
        print(f" EN,PID: {process.pid}")
        print(f"\n EN:")
        print(f"{'='*60}")
        
        # EN
        validation_metrics = {}
        
        # EN
        try:
            for line in process.stdout:
                line = line.strip()
                if line:
                    # EN
                    print(f" {line}")
                    job.add_log(line)
                    
                    # EN
                    if '%' in line and ('|' in line or '/' in line):
                        try:
                            # EN
                            percent_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                            if percent_match:
                                progress_percent = float(percent_match.group(1))
                                # EN50%EN100%(EN50%,EN50%)
                                job.progress = 50 + (progress_percent / 2)
                                print(f" EN: {job.progress:.1f}% (EN: {progress_percent:.1f}%)")
                                job.add_log(f"EN: {progress_percent:.1f}%")
                        except Exception as e:
                            job.add_log(f"EN: {e}")
                    
                    # EN
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
                                    print(f" EN: {acc_value:.4f}")
                                    job.add_log(f"EN: {acc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENaccuracyEN: {e}")
                    
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
                                    print(f" ENF1: {f1_value:.4f}")
                                    job.add_log(f"ENF1: {f1_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENF1EN: {e}")
                    
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
                                    print(f" EN: {prec_value:.4f}")
                                    job.add_log(f"EN: {prec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENprecisionEN: {e}")
                    
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
                                    print(f" EN: {rec_value:.4f}")
                                    job.add_log(f"EN: {rec_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENrecallEN: {e}")
                    
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
                                    print(f" ENAUC: {auc_value:.4f}")
                                    job.add_log(f"ENAUC: {auc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENAUCEN: {e}")
                    
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
                                    print(f" ENPR_AUC: {pr_auc_value:.4f}")
                                    job.add_log(f"ENPR_AUC: {pr_auc_value:.4f}")
                                    break
                        except Exception as e:
                            job.add_log(f"ENPR_AUCEN: {e}")
                    
                    # EN(tabulateEN)
                    if "Class" in line and "F1" in line and "Acc" in line:
                        job.add_log(" EN:")
                        print(" EN:")
                    
                    # EN,EN
                    if "---" in line or ("|" in line and any(word in line for word in ["F1", "Acc", "Prec", "Rec"])):
                        # EN,EN
                        pass
                    
                    # EN
                    if any(error_word in line.lower() for error_word in ['error', 'exception', 'failed', 'traceback']):
                        print(f" EN: {line}")
                        job.add_log(f"EN: {line}")
        
        except Exception as e:
            error_msg = f"EN: {e}"
            job.add_log(error_msg)
            print(f" {error_msg}")
        
        print(f"{'='*60}")
        print(f" EN...")
        
        # EN
        return_code = process.wait()
        print(f" EN,EN: {return_code}")
        
        # EN100%
        job.progress = 100
        
        # EN
        has_metrics = any(key.startswith('validation_') for key in job.metrics.keys())
        
        if return_code == 0 or has_metrics:
            job.status = "completed"
            job.current_phase = "completed"  # EN
            job.add_log(" EN!")
            print(f" EN!")
            
            # EN+EN
            print(f"\n EN+EN:")
            print(f"{'='*80}")
            job.add_log("\n EN+EN:")
            job.add_log("="*80)
            
            # EN
            if validation_metrics:
                validation_info = "\n EN:"
                print(validation_info)
                job.add_log(validation_info)
                
                for key, value in validation_metrics.items():
                    if isinstance(value, (int, float)):
                        metric_line = f"• {key.replace('validation_', '')}: {value:.4f}"
                        if value <= 1.0:  # EN
                            metric_line += f" ({value*100:.1f}%)"
                    else:
                        metric_line = f"• {key.replace('validation_', '')}: {value}"
                    print(f"  {metric_line}")
                    job.add_log(f"  {metric_line}")
            
            # EN
            if validation_metrics.get("validation_accuracy", 0) > 0:
                acc = validation_metrics["validation_accuracy"]
                assessment = "\n EN:"
                if acc > 0.9:
                    assessment += "\n   EN!EN90%,EN"
                elif acc > 0.8:
                    assessment += "\n   EN!EN80%,EN"
                elif acc > 0.7:
                    assessment += "\n   EN!EN70-80%EN,EN"
                else:
                    assessment += "\n   EN!EN70%,EN"
                
                print(assessment)
                job.add_log(assessment)
            
            # EN
            next_steps = f"\n EN:\n• 1. EN: {job.get_log_file_path()}\n• 2. EN\n• 3. EN\n• 4. EN"
            print(next_steps)
            job.add_log(next_steps)
            
            print(f"{'='*80}")
            job.add_log("="*80)
            
        else:
            job.status = "failed"
            job.current_phase = "completed"  # EN
            failure_msg = f" EN,EN: {return_code}"
            job.add_log(failure_msg)
            print(failure_msg)
        
        print(f" vulcan EN+EN")
        print(f"{'='*80}\n")
        
    except Exception as e:
        job.add_log(f" EN: {str(e)}")
        print(f" EN: {str(e)}")
        logger.error(f"EN: {e}")
        job.current_phase = "completed"  # EN,EN
        job.status = "completed"  # EN,EN

if __name__ == '__main__':
    print("=" * 60)
    print(" vulcan-Detection Backend Server")
    print("=" * 60)
    print(" ENAPIEN:")
    print("  GET  /api/health                    - EN")
    print("  GET  /api/models                    - EN")
    print("  GET  /api/datasets                  - EN")
    print("  POST /api/generate-config           - EN")
    print("  POST /api/generate-config-from-template - EN")
    print("  POST /api/start-training            - EN")
    print("  GET  /api/training-status/<job_id>  - EN")
    print("  GET  /api/training-logs/<job_id>    - EN")
    print("  POST /api/start-validation/<config_name> - EN")
    print("  GET  /api/validation-status/<job_id> - EN")
    print("  GET  /api/test-validation          - EN")
    print("  POST /api/start-dataset-optimization - EN")
    print("  GET  /api/optimization-status/<job_id> - EN")
    print("  GET  /api/optimization-logs/<job_id> - EN")
    print("  POST /api/start-paper-search       - EN")
    print("  GET  /api/paper-search-status/<job_id> - EN")
    print("  GET  /api/paper-search-logs/<job_id> - EN")
    print("  GET  /api/papers/<job_id>          - EN")
    print("  POST /api/start-data-collection    - EN")
    print("  GET  /api/data-collection-status/<job_id> - EN")
    print("  GET  /api/data-collection-logs/<job_id> - EN")
    print("  GET  /api/data-collection-results/<job_id> - EN")
    print("=" * 60)
    
    # EN
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # EN
    import sys
    import os
    
    # EN
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # EN
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print(" EN,EN")
    print(" ENFlaskEN...")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False) 