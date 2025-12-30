#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vulcan-Detection 配置模板管理系统
提供各种模型的配置模板和自动配置生成功能
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

class ConfigTemplate:
    """配置模板类"""
    
    def __init__(self, name: str, template: Dict[str, Any], description: str = ""):
        self.name = name
        self.template = template
        self.description = description
    
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """基于模板生成配置"""
        config = self._deep_copy_dict(self.template)
        
        # 应用参数覆盖
        for key, value in kwargs.items():
            self._set_nested_value(config, key, value)
        
        return config
    
    def _deep_copy_dict(self, d: Dict) -> Dict:
        """深拷贝字典"""
        if isinstance(d, dict):
            return {k: self._deep_copy_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._deep_copy_dict(v) for v in d]
        else:
            return d
    
    def _set_nested_value(self, config: Dict, key: str, value: Any):
        """设置嵌套键值"""
        if '.' in key:
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            config[key] = value

class ConfigTemplateManager:
    """配置模板管理器"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """初始化所有模板"""
        
        # DeepWuKong 模板
        deepwukong_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'DeepWuKong',
                'BACKBONE': '',
                'PARAMS': {
                    'gnn': {
                        'name': 'gcn',
                        'w2v_path': '/root/autodl-tmp/CWE119/w2v.wv',
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
                        'XFG_paths_json': '/root/autodl-tmp/CWE119/train.json'
                    }
                },
                'PREPROCESS': {
                    'ENABLE': False,
                    'COMPOSE': []
                }
            },
            'TRAIN': {
                'BATCH_SIZE': 32,
                'EPOCHS': 10,
                'EVAL_INTERVAL': 1,
                'AMP': False,
                'DDP': False
            },
            'EVAL': {
                'MODEL_PATH': '',
                'MSF': {
                    'ENABLE': False
                }
            },
            'LOSS': {
                'NAME': 'cross_entropy'
            },
            'OPTIMIZER': {
                'NAME': 'adamw',
                'LR': 0.001,
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'polynomial',
                'POWER': 0.9,
                'WARMUP': 10,
                'WARMUP_RATIO': 0.1
            }
        }
        
        # GNNReGVD 模板  
        gnnregvd_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'GNNReGVD',
                'BACKBONE': '',
                'PARAMS': {
                    'encoder': '',
                    'config': '',
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
                'NAME': 'ReGVD',
                'ROOT': '/path/to/reveal',
                'dataloader': 'normal',
                'PARAMS': {
                    'args': {
                        'train_data_file': '/path/to/train.jsonl',
                        'eval_data_file': '/path/to/valid.jsonl', 
                        'test_data_file': '/path/to/test.jsonl',
                        'block_size': 400
                    }
                },
                'PREPROCESS': {
                    'ENABLE': False,
                    'COMPOSE': []
                }
            },
            'TRAIN': {
                'BATCH_SIZE': 32,
                'EPOCHS': 10,
                'EVAL_INTERVAL': 1,
                'AMP': False,
                'DDP': False
            },
            'EVAL': {
                'MODEL_PATH': '',
                'MSF': {
                    'ENABLE': False
                }
            },
            'LOSS': {
                'NAME': 'cross_entropy'
            },
            'OPTIMIZER': {
                'NAME': 'adamw',
                'LR': 0.001,
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'polynomial',
                'POWER': 0.9,
                'WARMUP': 10,
                'WARMUP_RATIO': 0.1
            }
        }
        
        # Devign 模板
        devign_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'Devign',
                'BACKBONE': '',
                'PARAMS': {
                    'encoder': '',
                    'config': '',
                    'tokenizer': 'roberta',
                    'args': {
                        'config_name': '',
                        'model_name_or_path': 'microsoft/graphcodebert-base',
                        'tokenizer_name': 'microsoft/graphcodebert-base',
                        'block_size': 400,
                        'hidden_size': 768,
                        'num_classes': 2
                    }
                },
                'PRETRAINED': 'checkpoints/backbones/xx.pth'
            },
            'DATASET': {
                'NAME': 'Devign_Partial',
                'ROOT': '/path/to/devign',
                'dataloader': 'normal',
                'PARAMS': {
                    'args': {
                        'train_data_file': '/path/to/train.jsonl',
                        'eval_data_file': '/path/to/valid.jsonl',
                        'test_data_file': '/path/to/test.jsonl'
                    }
                },
                'PREPROCESS': {
                    'ENABLE': False,
                    'COMPOSE': []
                }
            },
            'TRAIN': {
                'BATCH_SIZE': 32,
                'EPOCHS': 10,
                'EVAL_INTERVAL': 1,
                'AMP': False,
                'DDP': False
            },
            'EVAL': {
                'MODEL_PATH': '',
                'MSF': {
                    'ENABLE': False
                }
            },
            'LOSS': {
                'NAME': 'cross_entropy'
            },
            'OPTIMIZER': {
                'NAME': 'adamw',
                'LR': 0.001,
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'polynomial',
                'POWER': 0.9,
                'WARMUP': 10,
                'WARMUP_RATIO': 0.1
            }
        }
        
        # IVDetect 模板
        ivdetect_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'IVDetect',
                'BACKBONE': '',
                'PARAMS': {
                    'args': {
                        'hidden_size': 128,
                        'num_classes': 2,
                        'learning_rate': 0.001
                    }
                },
                'PRETRAINED': 'checkpoints/backbones/xx.pth'
            },
            'DATASET': {
                'NAME': 'IVDDataset',
                'ROOT': '/path/to/ivdetect',
                'dataloader': 'normal',
                'PARAMS': {
                    'args': {
                        'dataset_path': '/path/to/dataset'
                    }
                },
                'PREPROCESS': {
                    'ENABLE': False,
                    'COMPOSE': []
                }
            },
            'TRAIN': {
                'BATCH_SIZE': 32,
                'EPOCHS': 10,
                'EVAL_INTERVAL': 1,
                'AMP': False,
                'DDP': False
            },
            'EVAL': {
                'MODEL_PATH': '',
                'MSF': {
                    'ENABLE': False
                }
            },
            'LOSS': {
                'NAME': 'cross_entropy'
            },
            'OPTIMIZER': {
                'NAME': 'adamw',
                'LR': 0.001,
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'polynomial',
                'POWER': 0.9,
                'WARMUP': 10,
                'WARMUP_RATIO': 0.1
            }
        }
        
        # LineVul 模板
        linevul_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'LineVul',
                'BACKBONE': '',
                'PARAMS': {
                    'encoder': '',
                    'config': '',
                    'tokenizer': 'roberta',
                    'args': {
                        'config_name': '',
                        'model_name_or_path': 'microsoft/codebert-base',
                        'tokenizer_name': 'microsoft/codebert-base',
                        'block_size': 512,
                        'hidden_size': 768,
                        'num_classes': 2
                    }
                },
                'PRETRAINED': 'checkpoints/backbones/xx.pth'
            },
            'DATASET': {
                'NAME': 'LineVul',
                'ROOT': '/path/to/linevul',
                'dataloader': 'normal',
                'PARAMS': {
                    'args': {
                        'train_data_file': '/path/to/train.jsonl',
                        'eval_data_file': '/path/to/valid.jsonl',
                        'test_data_file': '/path/to/test.jsonl'
                    }
                },
                'PREPROCESS': {
                    'ENABLE': False,
                    'COMPOSE': []
                }
            },
            'TRAIN': {
                'BATCH_SIZE': 32,
                'EPOCHS': 10,
                'EVAL_INTERVAL': 1,
                'AMP': False,
                'DDP': False
            },
            'EVAL': {
                'MODEL_PATH': '',
                'MSF': {
                    'ENABLE': False
                }
            },
            'LOSS': {
                'NAME': 'cross_entropy'
            },
            'OPTIMIZER': {
                'NAME': 'adamw',
                'LR': 0.001,
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'polynomial',
                'POWER': 0.9,
                'WARMUP': 10,
                'WARMUP_RATIO': 0.1
            }
        }
        
        # VulBERTa 模板
        vulberta_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'VulBERTa_CNN',
                'BACKBONE': '',
                'PARAMS': {
                    'encoder': '',
                    'config': '',
                    'tokenizer': 'roberta',
                    'args': {
                        'config_name': '',
                        'model_name_or_path': 'huggingface/CodeBERTa-small-v1',
                        'tokenizer_name': 'huggingface/CodeBERTa-small-v1',
                        'block_size': 512,
                        'hidden_size': 768,
                        'num_classes': 2
                    }
                },
                'PRETRAINED': 'checkpoints/backbones/xx.pth'
            },
            'DATASET': {
                'NAME': 'VDdata',
                'ROOT': '/path/to/vulberta',
                'dataloader': 'normal',
                'PARAMS': {
                    'args': {
                        'train_data_file': '/path/to/train.jsonl',
                        'eval_data_file': '/path/to/valid.jsonl',
                        'test_data_file': '/path/to/test.jsonl'
                    }
                },
                'PREPROCESS': {
                    'ENABLE': False,
                    'COMPOSE': []
                }
            },
            'TRAIN': {
                'BATCH_SIZE': 32,
                'EPOCHS': 10,
                'EVAL_INTERVAL': 1,
                'AMP': False,
                'DDP': False
            },
            'EVAL': {
                'MODEL_PATH': '',
                'MSF': {
                    'ENABLE': False
                }
            },
            'LOSS': {
                'NAME': 'cross_entropy'
            },
            'OPTIMIZER': {
                'NAME': 'adamw',
                'LR': 0.001,
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'polynomial',
                'POWER': 0.9,
                'WARMUP': 10,
                'WARMUP_RATIO': 0.1
            }
        }
        
        # 注册所有模板
        self.templates['deepwukong'] = ConfigTemplate('DeepWuKong', deepwukong_template, 'DeepWuKong 图神经网络漏洞检测模型')
        self.templates['gnnregvd'] = ConfigTemplate('GNNReGVD', gnnregvd_template, 'GNN-based 回归漏洞检测模型')
        self.templates['devign'] = ConfigTemplate('Devign', devign_template, 'Devign 图嵌入漏洞检测模型')
        self.templates['ivdetect'] = ConfigTemplate('IVDetect', ivdetect_template, 'IVDetect 智能漏洞检测模型')
        self.templates['linevul'] = ConfigTemplate('LineVul', linevul_template, 'LineVul 代码行级别漏洞检测')
        self.templates['vulberta'] = ConfigTemplate('VulBERTa', vulberta_template, 'VulBERTa BERT变体漏洞检测')
    
    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """获取指定模板"""
        return self.templates.get(name.lower())
    
    def list_templates(self) -> List[str]:
        """列出所有可用模板"""
        return list(self.templates.keys())
    
    def list_models(self) -> List[str]:
        """列出所有可用模型"""
        return [template.template['MODEL']['NAME'] for template in self.templates.values()]
    
    def get_template_by_model(self, model_name: str) -> Optional[ConfigTemplate]:
        """根据模型名称获取模板"""
        model_map = {
            'deepwukong': 'deepwukong',
            'gnnregvd': 'gnnregvd', 
            'devign': 'devign',
            'ivdetect': 'ivdetect',
            'linevul': 'linevul',
            'vulberta_cnn': 'vulberta',
            'vulberta': 'vulberta'
        }
        
        template_key = model_map.get(model_name.lower())
        if template_key:
            return self.templates.get(template_key)
        return None
    
    def generate_config(self, model_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """生成指定模型的配置"""
        template = self.get_template_by_model(model_name)
        if template:
            return template.generate_config(**kwargs)
        return None 