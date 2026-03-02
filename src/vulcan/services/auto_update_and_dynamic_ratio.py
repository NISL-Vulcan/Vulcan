import os
import subprocess
import uuid
import time
import json
import hashlib
import random
import yaml
import re
from typing import List, Dict, Any, Tuple, Optional
import sys

class DynamicRatioTuner:
    def __init__(self):
        self._used_vuln_indices = set()
        self._vuln_shuffle_order = []
        self._last_sample_content = None
        self._last_sample_hash = None
        
        # ENOSEN
        self.dataset_dir = self._find_dataset_directory()
        self.configs_dir = self._find_configs_directory()
        
    def _find_dataset_directory(self) -> str:
        """ENOSENdatasetEN"""
        possible_paths = [
            './vulcan-Detection/dataset',
            './dataset',
            '../dataset',
            '../../dataset',
            '../../vulcan-Detection/dataset'
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                print(f"[OSEN] ENdatasetEN: {os.path.abspath(path)}")
                return path
                
        # EN,EN(EN)
        try:
            current_dir = os.getcwd()
            for item in os.listdir(current_dir):
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path) and 'dataset' in item.lower():
                    print(f"[OSEN] ENdataset: {os.path.abspath(item_path)}")
                    return item_path
        except Exception as e:
            print(f"[OSEN] EN: {e}")
                
        raise FileNotFoundError("ENdatasetEN")
    
    def _find_configs_directory(self) -> str:
        """ENOSENconfigsEN"""
        possible_paths = [
            './vulcan-Detection/configs',
            './configs',
            '../configs',
            '../../configs',
            '../../vulcan-Detection/configs'
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                print(f"[OSEN] ENconfigsEN: {os.path.abspath(path)}")
                return path
                
        # EN,EN(EN)
        try:
            current_dir = os.getcwd()
            for item in os.listdir(current_dir):
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path) and 'configs' in item.lower():
                    print(f"[OSEN] ENconfigs: {os.path.abspath(item_path)}")
                    return item_path
        except Exception as e:
            print(f"[OSEN] EN: {e}")
                
        raise FileNotFoundError("ENconfigsEN")
    
    def _find_data_files(self) -> Tuple[str, str]:
        """EN"""
        vuln_files = []
        nonvuln_files = []
        
        # ENdatasetEN
        for file in os.listdir(self.dataset_dir):
            file_path = os.path.join(self.dataset_dir, file)
            if os.path.isfile(file_path) and file.endswith('.jsonl'):
                if 'vulnerable' in file.lower() or 'vuln' in file.lower():
                    vuln_files.append(file_path)
                elif 'non' in file.lower() and 'vulnerable' in file.lower():
                    nonvuln_files.append(file_path)
        
        # EN,EN
        if not vuln_files:
            for file in os.listdir(self.dataset_dir):
                file_path = os.path.join(self.dataset_dir, file)
                if os.path.isfile(file_path) and file.endswith('.jsonl'):
                    if 'vulnerables' in file:
                        vuln_files.append(file_path)
        
        if not nonvuln_files:
            for file in os.listdir(self.dataset_dir):
                file_path = os.path.join(self.dataset_dir, file)
                if os.path.isfile(file_path) and file.endswith('.jsonl'):
                    if 'non-vulnerables' in file:
                        nonvuln_files.append(file_path)
        
        if not vuln_files:
            raise FileNotFoundError(f"EN{self.dataset_dir}EN")
        if not nonvuln_files:
            raise FileNotFoundError(f"EN{self.dataset_dir}EN")
            
        vuln_path = vuln_files[0]  # EN
        nonvuln_path = nonvuln_files[0]
        
        print(f"[OSEN] EN: {vuln_path}")
        print(f"[OSEN] EN: {nonvuln_path}")
        
        return vuln_path, nonvuln_path
    
    def _find_config_file(self) -> str:
        """EN"""
        config_files = []
        
        # ENconfigsENyamlEN
        for file in os.listdir(self.configs_dir):
            file_path = os.path.join(self.configs_dir, file)
            if os.path.isfile(file_path) and file.endswith('.yaml'):
                config_files.append(file_path)
        
        if not config_files:
            raise FileNotFoundError(f"EN{self.configs_dir}ENyamlEN")
        
        # EN
        preferred_configs = []
        for config in config_files:
            if 'regvd' in config.lower() or 'reveal' in config.lower():
                preferred_configs.append(config)
        
        if preferred_configs:
            config_path = preferred_configs[0]
        else:
            config_path = config_files[0]
            
        print(f"[OSEN] EN: {config_path}")
        return config_path
    
    def _find_train_script(self) -> str:
        """EN"""
        possible_paths = [
            'vulcan-train',
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                print(f"[OSEN] EN: {os.path.abspath(path)}")
                return path
                
        # EN,EN
        for root, dirs, files in os.walk('.'):
            if 'train.py' in files:
                train_path = os.path.join(root, 'train.py')
                print(f"[OSEN] EN: {os.path.abspath(train_path)}")
                return train_path
                
        raise FileNotFoundError("EN(EN,EN vulcan-train EN)")

    def sample_and_generate_trainset(self, vuln_path: str, nonvuln_path: str, ratio: float, output_path: str):
        """EN"""
        # EN
        vuln_path_new = os.path.join(self.dataset_dir, 'vulnerables_new.jsonl')
        all_vuln_lines = []
        
        with open(vuln_path, 'r', encoding='utf-8') as f:
            all_vuln_lines.extend(f.readlines())
            
        if os.path.exists(vuln_path_new):
            with open(vuln_path_new, 'r', encoding='utf-8') as f:
                all_vuln_lines.extend(f.readlines())
                
        with open(nonvuln_path, 'r', encoding='utf-8') as f:
            nonvuln_lines = f.readlines()
            
        print(f"EN: {len(all_vuln_lines)}, EN: {len(nonvuln_lines)}")

        # 1. EN450~550EN
        num_vuln = random.randint(450, 550)
        if num_vuln > len(all_vuln_lines):
            num_vuln = len(all_vuln_lines)

        # 2. EN
        if not hasattr(self, '_used_vuln_indices') or len(self._used_vuln_indices) >= len(all_vuln_lines):
            self._used_vuln_indices = set()
            self._vuln_shuffle_order = list(range(len(all_vuln_lines)))
            random.shuffle(self._vuln_shuffle_order)
            
        # EN
        available_indices = [i for i in self._vuln_shuffle_order if i not in self._used_vuln_indices]
        if len(available_indices) < num_vuln:
            # EN,EN
            self._used_vuln_indices = set()
            self._vuln_shuffle_order = list(range(len(all_vuln_lines)))
            random.shuffle(self._vuln_shuffle_order)
            available_indices = self._vuln_shuffle_order
            
        chosen_indices = available_indices[:num_vuln]
        for idx in chosen_indices:
            self._used_vuln_indices.add(idx)
        vuln_sample = [all_vuln_lines[i] for i in chosen_indices]

        # 3. EN0.1~0.9EN,EN=EN/ratio
        ratio = max(0.1, min(0.9, ratio))
        num_nonvuln = int(round(num_vuln / ratio))
        num_nonvuln = min(num_nonvuln, len(nonvuln_lines))
        random.shuffle(nonvuln_lines)
        nonvuln_sample = nonvuln_lines[:num_nonvuln]

        with open(output_path, 'w', encoding='utf-8') as f:
            for line in vuln_sample + nonvuln_sample:
                f.write(line)
                
        actual_ratio = len(vuln_sample) / len(nonvuln_sample) if len(nonvuln_sample) > 0 else 0
        print(f"[EN] EN: {output_path}")
        print(f"[EN] EN: {len(vuln_sample)} (EN: 450~550, EN: {len(vuln_sample)})")
        print(f"[EN] EN: {len(nonvuln_sample)} (EN: {ratio:.3f}:1, EN: {actual_ratio:.3f}:1)")
        print(f"[EN] EN: {len(vuln_sample) + len(nonvuln_sample)}")

        # EN
        with open(output_path, 'r', encoding='utf-8') as f:
            current_content = f.readlines()
        current_hash = hashlib.md5(''.join(current_content).encode('utf-8')).hexdigest()
        
        if self._last_sample_content is not None:
            diff_count = sum(1 for a, b in zip(self._last_sample_content, current_content) if a != b)
            print(f"[EN] EN: {diff_count}")
            if current_hash == self._last_sample_hash:
                print("[EN] EN!")
            else:
                print("[EN] EN.")
                
        self._last_sample_content = current_content
        self._last_sample_hash = current_hash

    def evaluate_ratio(self, ratio: float) -> Tuple[float, Dict[str, float]]:
        """
        EN,EN(ENF1-score)EN.
        EN,EN.
        """
        # ENOSEN
        vuln_path, nonvuln_path = self._find_data_files()
        config_path = self._find_config_file()
        train_py = self._find_train_script()
        
        # 1. EN(EN)
        tmp_id = str(uuid.uuid4())[:8]
        trainset_path = os.path.join(self.dataset_dir, f'train_dynamic_{tmp_id}_{ratio:.3f}.jsonl')
        self.sample_and_generate_trainset(vuln_path, nonvuln_path, ratio, trainset_path)
        
        # 2. EN yaml EN train_data_file EN,EN
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        cfg['DATASET']['PARAMS']['args']['train_data_file'] = trainset_path
        
        tmp_cfg_path = os.path.join(self.configs_dir, f'tmp_dynamic_{tmp_id}_{ratio:.3f}.yaml')
        with open(tmp_cfg_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True)
            
        # 3. EN
        import sys
        cmd = [sys.executable, train_py, '--cfg', tmp_cfg_path]
        print(f"[EN] EN,EN={ratio:.3f},EN={trainset_path}")
        print(f"[EN] EN: {train_py}")
        print(f"[EN] EN: {tmp_cfg_path}")
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[EN] EN: {start_time}")
        
        metrics = {}
        f1 = 0.0
        try:
            # EN
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1,
                env=env,
                universal_newlines=True
            )
            
            patterns = {
                'F1': r'F1[-_ ]*score[::]?\s*([0-9.]+)',
                'Accuracy': r'Acc(?:uracy)?[::]?\s*([0-9.]+)',
                'Precision': r'Precision[::]?\s*([0-9.]+)',
                'Recall': r'Recall[::]?\s*([0-9.]+)',
                'ROC_AUC': r'ROC[_-]?AUC[::]?\s*([0-9.]+)',
                'PR_AUC': r'PR[_-]?AUC[::]?\s*([0-9.]+)'
            }
            
            # EN
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                    
                if line:
                    print(line.rstrip())  # EN
                    for key, pat in patterns.items():
                        m = re.search(pat, line, re.I)
                        if m:
                            metrics[key] = float(m.group(1))
                            if key == 'F1':
                                f1 = float(m.group(1))
                                print(f"[EN] ENF1EN: {f1:.4f}")
                            
            return_code = process.wait()
            end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print(f"[EN] EN: {end_time}")
            print(f"[EN] EN: {return_code}")
            
            if return_code != 0:
                print(f"[EN] EN: {return_code}")
                f1 = 0.0
                metrics = {}
            
        except Exception as e:
            print(f"[EN] EN: {e}")
            import traceback
            traceback.print_exc()
            f1 = 0.0
            metrics = {}
            
        # 4. EN
        try:
            if os.path.exists(trainset_path):
                os.remove(trainset_path)
                print(f"[EN] EN: {trainset_path}")
            if os.path.exists(tmp_cfg_path):
                os.remove(tmp_cfg_path)
                print(f"[EN] EN: {tmp_cfg_path}")
        except Exception as e:
            print(f"[EN] EN: {e}")
            
        return f1, metrics

    def get_best_ratio(self, max_iter: int = 15) -> float:
        """
        EN0.1EN0.9EN.
        ENmax_iter,EN.
        EN.
        """
        print("[EN] EN...")
        left, right = 0.1, 0.9
        eps = 0.01  # EN
        best_ratio = left
        best_score = -float('inf')
        best_metrics = {}
        iter_count = 0
        
        while right - left > eps and iter_count < max_iter:
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            
            print(f"\n[EN] EN{iter_count + 1}EN: EN {mid1:.3f} EN {mid2:.3f}")
            
            score1, metrics1 = self.evaluate_ratio(mid1)
            score2, metrics2 = self.evaluate_ratio(mid2)
            
            print(f"[EN] EN {mid1:.3f} ENF1EN: {score1:.4f}")
            print(f"[EN] EN {mid2:.3f} ENF1EN: {score2:.4f}")
            
            if score1 < score2:
                left = mid1
                if score2 > best_score:
                    best_score = score2
                    best_ratio = mid2
                    best_metrics = metrics2
            else:
                right = mid2
                if score1 > best_score:
                    best_score = score1
                    best_ratio = mid1
                    best_metrics = metrics1
                    
            iter_count += 1
            print(f"[EN] EN: {best_ratio:.3f}, F1EN: {best_score:.4f}")
            print(f"[EN] EN: {int((iter_count / max_iter) * 100)}%")
            
        print(f"\n[EN] EN,EN: {best_ratio:.3f},EN{iter_count}EN")
        print(f"[EN]")
        for k, v in best_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        return round(best_ratio, 3)

if __name__ == "__main__":
    print("\n==== EN(EN) ====")
    
    try:
        print("EN...")
        tuner = DynamicRatioTuner()
        print("EN")
        
        print("EN...")
        best_ratio = tuner.get_best_ratio()
        print(f"\nEN: {best_ratio}")
        
    except FileNotFoundError as e:
        print(f"EN: {e}")
        print("EN")
        sys.exit(1)
    except Exception as e:
        print(f"EN: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 