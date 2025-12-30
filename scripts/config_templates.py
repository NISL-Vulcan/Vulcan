#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vulcan-Detection 配置文件模板系统
提供预定义的配置模板和智能配置生成
"""

import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path

class ConfigTemplate:
    """配置模板类"""
    
    def __init__(self, name: str, template: Dict[str, Any], description: str = ""):
        self.name = name
        self.template = template
        self.description = description
    
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """根据参数生成配置"""
        config = self._deep_copy_dict(self.template)
        self._update_config_with_params(config, **kwargs)
        return config
    
    def _deep_copy_dict(self, d: Dict) -> Dict:
        """深度复制字典"""
        import copy
        return copy.deepcopy(d)
    
    def _update_config_with_params(self, config: Dict[str, Any], **kwargs):
        """根据参数更新配置"""
        # 更新基本参数
        if 'device' in kwargs:
            config['DEVICE'] = kwargs['device']
        
        if 'save_dir' in kwargs:
            config['SAVE_DIR'] = kwargs['save_dir']
        
        # 更新训练参数
        if 'TRAIN' in config:
            train_params = ['batch_size', 'epochs', 'eval_interval']
            for param in train_params:
                if param in kwargs:
                    param_key = param.upper()
                    if param == 'batch_size':
                        param_key = 'BATCH_SIZE'
                    elif param == 'epochs':
                        param_key = 'EPOCHS'
                    elif param == 'eval_interval':
                        param_key = 'EVAL_INTERVAL'
                    config['TRAIN'][param_key] = kwargs[param]
        
        # 更新优化器参数
        if 'OPTIMIZER' in config and 'learning_rate' in kwargs:
            config['OPTIMIZER']['LR'] = kwargs['learning_rate']
        
        # 更新数据集路径
        if 'DATASET' in config and 'data_path' in kwargs:
            config['DATASET']['ROOT'] = kwargs['data_path']

class ConfigTemplateManager:
    """配置模板管理器"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """初始化预定义模板"""
        
        # DeepWuKong + DWK_Dataset 模板
        deepwukong_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'DeepWuKong',
                'BACKBONE': None,
                'PARAMS': {
                    'gnn': {
                        'name': 'gcn',
                        'w2v_path': '/path/to/w2v.wv',
                        'embed_size': 256,
                        'hidden_size': 256,
                        'pooling_ratio': 0.8,
                        'drop_out': 0.5,
                        'n_hidden_layers': 3,
                        'n_head': 3,
                        'n_gru': 3,
                        'edge_sample_ratio': 0.8,
                        'rnn': {
                            'hidden_size': 256,
                            'num_layers': 1,
                            'drop_out': 0.5,
                            'use_bi': True,
                            'activation': 'relu'
                        }
                    },
                    'classifier': {
                        'hidden_size': 512,
                        'n_hidden_layers': 2,
                        'n_classes': 2,
                        'drop_out': 0.5
                    }
                },
                'PRETRAINED': 'checkpoints/backbones/xx.pth'
            },
            'DATASET': {
                'NAME': 'DWK_Dataset',
                'ROOT': '',
                'dataloader': 'geometric',
                'PARAMS': {
                    'args': {
                        'XFG_paths_json': '/path/to/train.json',
                        'w2v_path': '/path/to/w2v.wv',
                        'token_max_parts': 16
                    }
                },
                'PREPROCESS': {
                    'ENABLE': False,
                    'COMPOSE': ['Normalize', 'PadSequence', 'OneHotEncode']
                }
            },
            'TRAIN': {
                'INPUT_SIZE': 128,
                'BATCH_SIZE': 128,
                'EPOCHS': 20,
                'EVAL_INTERVAL': 50,
                'AMP': False,
                'DDP': False
            },
            'LOSS': {
                'NAME': 'CrossEntropy',
                'CLS_WEIGHTS': False
            },
            'OPTIMIZER': {
                'NAME': 'adamw',
                'LR': 0.001,
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'warmuppolylr',
                'POWER': 0.9,
                'WARMUP': 10,
                'WARMUP_RATIO': 0.1
            },
            'EVAL': {
                'INPUT_SIZE': 128,
                'MODEL_PATH': 'checkpoints/pretrained/ade.pth',
                'MSF': {
                    'ENABLE': False,
                    'FLIP': True,
                    'SCALES': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
                }
            },
            'TEST': {
                'MODEL_PATH': 'checkpoints/pretrained/ade.pth',
                'FILE': 'assests/ade',
                'INPUT_SIZE': 128,
                'OVERLAY': True
            }
        }
        
        # Devign 模板
        devign_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'Devign',
                'BACKBONE': None,
                'PARAMS': {
                    'encoder': None,
                    'config': None,
                    'tokenizer': 'roberta',
                    'args': {
                        'config_name': '',
                        'gnn': 'ReGGNN',
                        'feature_dim_size': 768,
                        'hidden_size': 256,
                        'num_GNN_layers': 2,
                        'model_name_or_path': 'microsoft/graphcodebert-base',
                        'remove_residual': False,
                        'att_op': 'mul',
                        'num_classes': 2,
                        'format': 'uni',
                        'window_size': 5
                    }
                },
                'PRETRAINED': 'checkpoints/backbones/xx.pth'
            },
            'DATASET': {
                'NAME': 'Devign_Partial',
                'ROOT': '',
                'PARAMS': {
                    'input_path': '/path/to/devign_partial_data/input'
                },
                'PREPROCESS': {
                    'ENABLE': False,
                    'COMPOSE': ['Normalize', 'PadSequence', 'OneHotEncode']
                }
            },
            'TRAIN': {
                'INPUT_SIZE': 128,
                'BATCH_SIZE': 8,
                'EPOCHS': 50,
                'EVAL_INTERVAL': 50,
                'AMP': False,
                'DDP': False
            },
            'LOSS': {
                'NAME': 'CrossEntropy',
                'CLS_WEIGHTS': False
            },
            'OPTIMIZER': {
                'NAME': 'adamw',
                'LR': 0.001,
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'warmuppolylr',
                'POWER': 0.9,
                'WARMUP': 1,
                'WARMUP_RATIO': 0.1
            },
            'EVAL': {
                'INPUT_SIZE': 128,
                'MODEL_PATH': 'checkpoints/pretrained/vuln.pth',
                'MSF': {
                    'ENABLE': False,
                    'FLIP': True,
                    'SCALES': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
                }
            },
            'TEST': {
                'MODEL_PATH': 'checkpoints/pretrained/vuln.pth',
                'FILE': 'assests/ade',
                'INPUT_SIZE': 128,
                'OVERLAY': True
            }
        }
        
        # IVDetect 模板
        ivdetect_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'IVDetect',
                'BACKBONE': None,
                'PARAMS': {
                    'hidden_size': 128,
                    'num_node_feature': 5,
                    'num_classes': 2,
                    'feature_representation_size': 128,
                    'drop_out_rate': 0.3,
                    'num_conv_layers': 3
                },
                'PRETRAINED': None
            },
            'DATASET': {
                'NAME': 'IVDetect_Dataset',
                'ROOT': '',
                'PARAMS': {
                    'processed_dir': 'data/',
                    'train_path': 'data/train_graph/',
                    'test_path': 'data/test_graph/',
                    'valid_path': 'data/valid_graph/'
                },
                'PREPROCESS': {
                    'ENABLE': False,
                    'COMPOSE': ['GraphPreprocess']
                }
            },
            'TRAIN': {
                'INPUT_SIZE': 128,
                'BATCH_SIZE': 1,
                'EPOCHS': 100,
                'EVAL_INTERVAL': 10,
                'AMP': False,
                'DDP': False
            },
            'LOSS': {
                'NAME': 'CrossEntropy',
                'CLS_WEIGHTS': False
            },
            'OPTIMIZER': {
                'NAME': 'adam',
                'LR': 0.0001,
                'WEIGHT_DECAY': 0.0
            },
            'SCHEDULER': {
                'NAME': 'none',
                'POWER': 1.0,
                'WARMUP': 0,
                'WARMUP_RATIO': 0.0
            },
            'EVAL': {
                'INPUT_SIZE': 128,
                'MODEL_PATH': 'model/trained_model.pt',
                'MSF': {
                    'ENABLE': False,
                    'FLIP': False,
                    'SCALES': [1.0]
                }
            },
            'TEST': {
                'MODEL_PATH': 'model/trained_model.pt',
                'FILE': 'test_data',
                'INPUT_SIZE': 128,
                'OVERLAY': False
            }
        }
        
        # LineVul 模板
        linevul_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'LineVul',
                'BACKBONE': None,
                'PARAMS': {
                    'model_type': 'roberta',
                    'tokenizer_name': 'microsoft/codebert-base',
                    'model_name_or_path': 'microsoft/codebert-base',
                    'block_size': 512,
                    'hidden_size': 768,
                    'num_classes': 2
                },
                'PRETRAINED': None
            },
            'DATASET': {
                'NAME': 'LineVul_Dataset',
                'ROOT': '',
                'PARAMS': {
                    'train_data_file': 'dataset/train.jsonl',
                    'eval_data_file': 'dataset/valid.jsonl',
                    'test_data_file': 'dataset/test.jsonl'
                },
                'PREPROCESS': {
                    'ENABLE': True,
                    'COMPOSE': ['Tokenize', 'PadSequence']
                }
            },
            'TRAIN': {
                'INPUT_SIZE': 512,
                'BATCH_SIZE': 16,
                'EPOCHS': 10,
                'EVAL_INTERVAL': 1,
                'AMP': True,
                'DDP': False
            },
            'LOSS': {
                'NAME': 'CrossEntropy',
                'CLS_WEIGHTS': False
            },
            'OPTIMIZER': {
                'NAME': 'adamw',
                'LR': 2e-5,
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'linear',
                'POWER': 1.0,
                'WARMUP': 0.1,
                'WARMUP_RATIO': 0.1
            },
            'EVAL': {
                'INPUT_SIZE': 512,
                'MODEL_PATH': 'checkpoints/linevul.pth',
                'MSF': {
                    'ENABLE': False,
                    'FLIP': False,
                    'SCALES': [1.0]
                }
            },
            'TEST': {
                'MODEL_PATH': 'checkpoints/linevul.pth',
                'FILE': 'test_data',
                'INPUT_SIZE': 512,
                'OVERLAY': False
            }
        }
        
        # 注册模板
        self.templates['deepwukong'] = ConfigTemplate(
            'DeepWuKong',
            deepwukong_template,
            '基于图神经网络的代码漏洞检测模型'
        )
        
        self.templates['devign'] = ConfigTemplate(
            'Devign',
            devign_template,
            '基于图嵌入的漏洞检测模型'
        )
        
        self.templates['ivdetect'] = ConfigTemplate(
            'IVDetect',
            ivdetect_template,
            '智能漏洞检测模型'
        )
        
        self.templates['linevul'] = ConfigTemplate(
            'LineVul',
            linevul_template,
            '基于代码行的漏洞检测模型'
        )
    
    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """获取模板"""
        return self.templates.get(name.lower())
    
    def list_templates(self) -> List[str]:
        """列出所有模板"""
        return list(self.templates.keys())
    
    def get_template_info(self) -> Dict[str, str]:
        """获取模板信息"""
        return {name: template.description for name, template in self.templates.items()}
    
    def generate_config_by_model(self, model_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """根据模型名称生成配置"""
        # 根据模型名称匹配最佳模板
        model_lower = model_name.lower()
        
        template_mapping = {
            'deepwukong': 'deepwukong',
            'devign': 'devign',
            'ivdetect': 'ivdetect',
            'linevul': 'linevul',
            'regvd': 'devign',  # ReGVD 使用 Devign 模板
            'vulberta': 'linevul',  # VulBERTa 使用 LineVul 模板
            'vuldeepecker': 'deepwukong'  # VulDeePecker 使用 DeepWuKong 模板
        }
        
        template_name = None
        for key, value in template_mapping.items():
            if key in model_lower:
                template_name = value
                break
        
        if not template_name:
            # 默认使用 DeepWuKong 模板
            template_name = 'deepwukong'
        
        template = self.get_template(template_name)
        if template:
            return template.generate_config(**kwargs)
        
        return None
    
    def save_config(self, config: Dict[str, Any], filepath: str) -> bool:
        """保存配置到文件"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            return True
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """验证配置文件"""
        errors = []
        
        # 检查必需的顶级键
        required_keys = ['DEVICE', 'SAVE_DIR', 'MODEL', 'DATASET', 'TRAIN', 'LOSS', 'OPTIMIZER']
        for key in required_keys:
            if key not in config:
                errors.append(f"缺少必需的配置项: {key}")
        
        # 检查MODEL配置
        if 'MODEL' in config:
            if 'NAME' not in config['MODEL']:
                errors.append("MODEL配置中缺少NAME字段")
        
        # 检查DATASET配置
        if 'DATASET' in config:
            if 'NAME' not in config['DATASET']:
                errors.append("DATASET配置中缺少NAME字段")
        
        # 检查TRAIN配置
        if 'TRAIN' in config:
            train_required = ['BATCH_SIZE', 'EPOCHS']
            for key in train_required:
                if key not in config['TRAIN']:
                    errors.append(f"TRAIN配置中缺少{key}字段")
        
        return len(errors) == 0, errors

# 使用示例
if __name__ == '__main__':
    manager = ConfigTemplateManager()
    
    print("可用的配置模板:")
    for name, desc in manager.get_template_info().items():
        print(f"  {name}: {desc}")
    
    print("\n生成DeepWuKong配置示例:")
    config = manager.generate_config_by_model(
        'DeepWuKong',
        device='cuda',
        batch_size=64,
        epochs=30,
        learning_rate=0.0005,
        save_dir='my_output'
    )
    
    if config:
        print(yaml.dump(config, default_flow_style=False, allow_unicode=True, indent=2))
        
        # 验证配置
        is_valid, errors = manager.validate_config(config)
        print(f"\n配置验证: {'通过' if is_valid else '失败'}")
        if errors:
            for error in errors:
                print(f"  - {error}") 