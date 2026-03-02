#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vulcan-Detection configuration template management.
Provides model templates and automatic config generation.
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

class ConfigTemplate:
    """Configuration template."""
    
    def __init__(self, name: str, template: Dict[str, Any], description: str = ""):
        self.name = name
        self.template = template
        self.description = description
    
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate config from template."""
        config = self._deep_copy_dict(self.template)
        
        # Apply overrides
        for key, value in kwargs.items():
            self._set_nested_value(config, key, value)
        
        return config
    
    def _deep_copy_dict(self, d: Dict) -> Dict:
        """Deep-copy a dictionary."""
        if isinstance(d, dict):
            return {k: self._deep_copy_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._deep_copy_dict(v) for v in d]
        else:
            return d
    
    def _set_nested_value(self, config: Dict, key: str, value: Any):
        """Set nested key value."""
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
    """Configuration template manager."""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all templates."""
        
        # DeepWuKong template
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
        
        # GNNReGVD template
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
        
        # Devign template
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
        
        # IVDetect template
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
        
        # LineVul template
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
        
        # VulBERTa template
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
        
        # TrVD template
        trvd_template = {
            'DEVICE': 'cuda',
            'SAVE_DIR': 'output',
            'MODEL': {
                'NAME': 'TrVD',
                'BACKBONE': '',
                'PARAMS': {
                    'embedding_dim': 128,
                    'hidden_dim': 100,
                    'vocab_size': 5000,
                    'encode_dim': 128,
                    'label_size': 2,
                    'n_heads': 4,
                    'n_transformer_layers': 2,
                    'dropout': 0.2,
                    'pretrained_weight_path': ''
                },
                'PRETRAINED': ''
            },
            'DATASET': {
                'NAME': 'TrVD',
                'ROOT': '',
                'dataloader': 'trvd',
                'PARAMS': {
                    'args': {
                        'dataset_path': '/path/to/trvd/dataset',
                        'language_lib': '',
                        'grammar_dir': '',
                        'w2v_path': '',
                        'embedding_size': 128,
                        'language': 'cpp',
                        'normalize': True,
                        'label_size': 2
                    }
                },
                'PREPROCESS': {
                    'ENABLE': False,
                    'COMPOSE': []
                }
            },
            'TRAIN': {
                'BATCH_SIZE': 32,
                'EPOCHS': 100,
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
                'NAME': 'CrossEntropy'
            },
            'OPTIMIZER': {
                'NAME': 'adamax',
                'LR': 0.001,
                'WEIGHT_DECAY': 0.01
            },
            'SCHEDULER': {
                'NAME': 'steplr',
                'POWER': 0.9,
                'WARMUP': 0,
                'WARMUP_RATIO': 0.1,
                'STEP_SIZE': 10,
                'GAMMA': 0.8
            }
        }

        # Register all templates
        self.templates['trvd'] = ConfigTemplate('TrVD', trvd_template, 'TrVD AST decomposition based vulnerability detection model')
        self.templates['deepwukong'] = ConfigTemplate('DeepWuKong', deepwukong_template, 'DeepWuKong graph neural vulnerability detection model')
        self.templates['gnnregvd'] = ConfigTemplate('GNNReGVD', gnnregvd_template, 'GNN-based regression vulnerability detection model')
        self.templates['devign'] = ConfigTemplate('Devign', devign_template, 'Devign graph embedding vulnerability detection model')
        self.templates['ivdetect'] = ConfigTemplate('IVDetect', ivdetect_template, 'IVDetect intelligent vulnerability detection model')
        self.templates['linevul'] = ConfigTemplate('LineVul', linevul_template, 'LineVul line-level vulnerability detection')
        self.templates['vulberta'] = ConfigTemplate('VulBERTa', vulberta_template, 'VulBERTa variant vulnerability detection')
    
    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """Get a specific template."""
        return self.templates.get(name.lower())
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        return list(self.templates.keys())
    
    def list_models(self) -> List[str]:
        """List all available models."""
        return [template.template['MODEL']['NAME'] for template in self.templates.values()]
    
    def get_template_by_model(self, model_name: str) -> Optional[ConfigTemplate]:
        """Get template by model name."""
        model_map = {
            'deepwukong': 'deepwukong',
            'gnnregvd': 'gnnregvd', 
            'devign': 'devign',
            'ivdetect': 'ivdetect',
            'linevul': 'linevul',
            'vulberta_cnn': 'vulberta',
            'vulberta': 'vulberta',
            'trvd': 'trvd',
        }
        
        template_key = model_map.get(model_name.lower())
        if template_key:
            return self.templates.get(template_key)
        return None
    
    def generate_config(self, model_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate config for the specified model."""
        template = self.get_template_by_model(model_name)
        if template:
            return template.generate_config(**kwargs)
        return None 