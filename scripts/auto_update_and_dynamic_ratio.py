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
        
        # 使用OS查找文件路径
        self.dataset_dir = self._find_dataset_directory()
        self.configs_dir = self._find_configs_directory()
        
    def _find_dataset_directory(self) -> str:
        """使用OS查找dataset目录"""
        possible_paths = [
            './vulcan-Detection/dataset',
            './dataset',
            '../dataset',
            '../../dataset',
            '../../vulcan-Detection/dataset'
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                print(f"[OS查找] 找到dataset目录: {os.path.abspath(path)}")
                return path
                
        # 如果没找到，尝试在当前目录的直接子目录中搜索（避免深度遍历）
        try:
            current_dir = os.getcwd()
            for item in os.listdir(current_dir):
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path) and 'dataset' in item.lower():
                    print(f"[OS查找] 在子目录中找到dataset: {os.path.abspath(item_path)}")
                    return item_path
        except Exception as e:
            print(f"[OS查找] 搜索子目录时出错: {e}")
                
        raise FileNotFoundError("无法找到dataset目录")
    
    def _find_configs_directory(self) -> str:
        """使用OS查找configs目录"""
        possible_paths = [
            './vulcan-Detection/configs',
            './configs',
            '../configs',
            '../../configs',
            '../../vulcan-Detection/configs'
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                print(f"[OS查找] 找到configs目录: {os.path.abspath(path)}")
                return path
                
        # 如果没找到，尝试在当前目录的直接子目录中搜索（避免深度遍历）
        try:
            current_dir = os.getcwd()
            for item in os.listdir(current_dir):
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path) and 'configs' in item.lower():
                    print(f"[OS查找] 在子目录中找到configs: {os.path.abspath(item_path)}")
                    return item_path
        except Exception as e:
            print(f"[OS查找] 搜索子目录时出错: {e}")
                
        raise FileNotFoundError("无法找到configs目录")
    
    def _find_data_files(self) -> Tuple[str, str]:
        """查找漏洞和非漏洞数据文件"""
        vuln_files = []
        nonvuln_files = []
        
        # 在dataset目录中查找文件
        for file in os.listdir(self.dataset_dir):
            file_path = os.path.join(self.dataset_dir, file)
            if os.path.isfile(file_path) and file.endswith('.jsonl'):
                if 'vulnerable' in file.lower() or 'vuln' in file.lower():
                    vuln_files.append(file_path)
                elif 'non' in file.lower() and 'vulnerable' in file.lower():
                    nonvuln_files.append(file_path)
        
        # 如果没有找到，尝试其他命名模式
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
            raise FileNotFoundError(f"在{self.dataset_dir}中未找到漏洞数据文件")
        if not nonvuln_files:
            raise FileNotFoundError(f"在{self.dataset_dir}中未找到非漏洞数据文件")
            
        vuln_path = vuln_files[0]  # 使用第一个找到的文件
        nonvuln_path = nonvuln_files[0]
        
        print(f"[OS查找] 漏洞数据文件: {vuln_path}")
        print(f"[OS查找] 非漏洞数据文件: {nonvuln_path}")
        
        return vuln_path, nonvuln_path
    
    def _find_config_file(self) -> str:
        """查找配置文件"""
        config_files = []
        
        # 在configs目录中查找yaml配置文件
        for file in os.listdir(self.configs_dir):
            file_path = os.path.join(self.configs_dir, file)
            if os.path.isfile(file_path) and file.endswith('.yaml'):
                config_files.append(file_path)
        
        if not config_files:
            raise FileNotFoundError(f"在{self.configs_dir}中未找到yaml配置文件")
        
        # 优先选择包含特定关键词的配置文件
        preferred_configs = []
        for config in config_files:
            if 'regvd' in config.lower() or 'reveal' in config.lower():
                preferred_configs.append(config)
        
        if preferred_configs:
            config_path = preferred_configs[0]
        else:
            config_path = config_files[0]
            
        print(f"[OS查找] 配置文件: {config_path}")
        return config_path
    
    def _find_train_script(self) -> str:
        """查找训练脚本"""
        possible_paths = [
            './vulcan-Detection/tools/train.py',
            './tools/train.py',
            '../tools/train.py',
            './train.py'
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                print(f"[OS查找] 训练脚本: {os.path.abspath(path)}")
                return path
                
        # 如果没找到，尝试在当前目录及其子目录中搜索
        for root, dirs, files in os.walk('.'):
            if 'train.py' in files:
                train_path = os.path.join(root, 'train.py')
                print(f"[OS查找] 在子目录中找到训练脚本: {os.path.abspath(train_path)}")
                return train_path
                
        raise FileNotFoundError("无法找到训练脚本train.py")

    def sample_and_generate_trainset(self, vuln_path: str, nonvuln_path: str, ratio: float, output_path: str):
        """采样并生成训练集"""
        # 合并原有和新生成的漏洞样本
        vuln_path_new = os.path.join(self.dataset_dir, 'vulnerables_new.jsonl')
        all_vuln_lines = []
        
        with open(vuln_path, 'r', encoding='utf-8') as f:
            all_vuln_lines.extend(f.readlines())
            
        if os.path.exists(vuln_path_new):
            with open(vuln_path_new, 'r', encoding='utf-8') as f:
                all_vuln_lines.extend(f.readlines())
                
        with open(nonvuln_path, 'r', encoding='utf-8') as f:
            nonvuln_lines = f.readlines()
            
        print(f"真实漏洞样本数: {len(all_vuln_lines)}, 真实非漏洞样本数: {len(nonvuln_lines)}")

        # 1. 每次采样漏洞样本数量在450~550之间浮动
        num_vuln = random.randint(450, 550)
        if num_vuln > len(all_vuln_lines):
            num_vuln = len(all_vuln_lines)

        # 2. 尽量不重复采样漏洞样本
        if not hasattr(self, '_used_vuln_indices') or len(self._used_vuln_indices) >= len(all_vuln_lines):
            self._used_vuln_indices = set()
            self._vuln_shuffle_order = list(range(len(all_vuln_lines)))
            random.shuffle(self._vuln_shuffle_order)
            
        # 选取未用过的索引
        available_indices = [i for i in self._vuln_shuffle_order if i not in self._used_vuln_indices]
        if len(available_indices) < num_vuln:
            # 剩余不够，重置再采样
            self._used_vuln_indices = set()
            self._vuln_shuffle_order = list(range(len(all_vuln_lines)))
            random.shuffle(self._vuln_shuffle_order)
            available_indices = self._vuln_shuffle_order
            
        chosen_indices = available_indices[:num_vuln]
        for idx in chosen_indices:
            self._used_vuln_indices.add(idx)
        vuln_sample = [all_vuln_lines[i] for i in chosen_indices]

        # 3. 比例在0.1~0.9之间，非漏洞样本数量=漏洞样本数/ratio
        ratio = max(0.1, min(0.9, ratio))
        num_nonvuln = int(round(num_vuln / ratio))
        num_nonvuln = min(num_nonvuln, len(nonvuln_lines))
        random.shuffle(nonvuln_lines)
        nonvuln_sample = nonvuln_lines[:num_nonvuln]

        with open(output_path, 'w', encoding='utf-8') as f:
            for line in vuln_sample + nonvuln_sample:
                f.write(line)
                
        actual_ratio = len(vuln_sample) / len(nonvuln_sample) if len(nonvuln_sample) > 0 else 0
        print(f"[采样] 生成训练集: {output_path}")
        print(f"[采样] 漏洞样本: {len(vuln_sample)} (目标: 450~550, 实际: {len(vuln_sample)})")
        print(f"[采样] 非漏洞样本: {len(nonvuln_sample)} (目标比例: {ratio:.3f}:1, 实际比例: {actual_ratio:.3f}:1)")
        print(f"[采样] 总样本数: {len(vuln_sample) + len(nonvuln_sample)}")

        # 自动分析与上一次采样文件的内容差异
        with open(output_path, 'r', encoding='utf-8') as f:
            current_content = f.readlines()
        current_hash = hashlib.md5(''.join(current_content).encode('utf-8')).hexdigest()
        
        if self._last_sample_content is not None:
            diff_count = sum(1 for a, b in zip(self._last_sample_content, current_content) if a != b)
            print(f"[分析] 与上一次采样文件不同的样本数: {diff_count}")
            if current_hash == self._last_sample_hash:
                print("[分析] 本次采样内容与上一次完全相同！")
            else:
                print("[分析] 本次采样内容与上一次有差异。")
                
        self._last_sample_content = current_content
        self._last_sample_hash = current_hash

    def evaluate_ratio(self, ratio: float) -> Tuple[float, Dict[str, float]]:
        """
        用真实数据集采样、训练并返回模型性能（如F1-score）和所有评估指标。
        采样数据和配置均为唯一临时文件，不影响原有训练和配置。
        """
        # 使用OS查找文件
        vuln_path, nonvuln_path = self._find_data_files()
        config_path = self._find_config_file()
        train_py = self._find_train_script()
        
        # 1. 采样生成新训练集（唯一临时文件）
        tmp_id = str(uuid.uuid4())[:8]
        trainset_path = os.path.join(self.dataset_dir, f'train_dynamic_{tmp_id}_{ratio:.3f}.jsonl')
        self.sample_and_generate_trainset(vuln_path, nonvuln_path, ratio, trainset_path)
        
        # 2. 修改 yaml 配置文件中的 train_data_file 路径，生成唯一临时配置
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        cfg['DATASET']['PARAMS']['args']['train_data_file'] = trainset_path
        
        tmp_cfg_path = os.path.join(self.configs_dir, f'tmp_dynamic_{tmp_id}_{ratio:.3f}.yaml')
        with open(tmp_cfg_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True)
            
        # 3. 实时读取训练输出并解析指标
        import sys
        cmd = [sys.executable, train_py, '--cfg', tmp_cfg_path]
        print(f"[训练] 启动训练，比例={ratio:.3f}，训练集={trainset_path}")
        print(f"[训练] 训练脚本: {train_py}")
        print(f"[训练] 配置文件: {tmp_cfg_path}")
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[训练] 开始时间: {start_time}")
        
        metrics = {}
        f1 = 0.0
        try:
            # 设置环境变量确保输出不被缓冲
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
                'F1': r'F1[-_ ]*score[:：]?\s*([0-9.]+)',
                'Accuracy': r'Acc(?:uracy)?[:：]?\s*([0-9.]+)',
                'Precision': r'Precision[:：]?\s*([0-9.]+)',
                'Recall': r'Recall[:：]?\s*([0-9.]+)',
                'ROC_AUC': r'ROC[_-]?AUC[:：]?\s*([0-9.]+)',
                'PR_AUC': r'PR[_-]?AUC[:：]?\s*([0-9.]+)'
            }
            
            # 实时读取输出
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                    
                if line:
                    print(line.rstrip())  # 实时输出到终端
                    for key, pat in patterns.items():
                        m = re.search(pat, line, re.I)
                        if m:
                            metrics[key] = float(m.group(1))
                            if key == 'F1':
                                f1 = float(m.group(1))
                                print(f"[指标] 检测到F1分数: {f1:.4f}")
                            
            return_code = process.wait()
            end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print(f"[训练] 结束时间: {end_time}")
            print(f"[训练] 返回码: {return_code}")
            
            if return_code != 0:
                print(f"[训练] 训练脚本返回非零退出码: {return_code}")
                f1 = 0.0
                metrics = {}
            
        except Exception as e:
            print(f"[训练] 训练失败: {e}")
            import traceback
            traceback.print_exc()
            f1 = 0.0
            metrics = {}
            
        # 4. 清理临时文件
        try:
            if os.path.exists(trainset_path):
                os.remove(trainset_path)
                print(f"[清理] 已删除临时训练集: {trainset_path}")
            if os.path.exists(tmp_cfg_path):
                os.remove(tmp_cfg_path)
                print(f"[清理] 已删除临时配置文件: {tmp_cfg_path}")
        except Exception as e:
            print(f"[清理] 临时文件删除失败: {e}")
            
        return f1, metrics

    def get_best_ratio(self, max_iter: int = 15) -> float:
        """
        使用二分法在0.1到0.9区间自动搜索最佳数据集比例。
        增加最大训练次数max_iter，防止死循环。
        最后输出最佳比例对应的所有评估指标。
        """
        print("[二分法] 开始自动搜索最佳数据集比例...")
        left, right = 0.1, 0.9
        eps = 0.01  # 精度
        best_ratio = left
        best_score = -float('inf')
        best_metrics = {}
        iter_count = 0
        
        while right - left > eps and iter_count < max_iter:
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            
            print(f"\n[二分法] 第{iter_count + 1}次迭代: 测试比例 {mid1:.3f} 和 {mid2:.3f}")
            
            score1, metrics1 = self.evaluate_ratio(mid1)
            score2, metrics2 = self.evaluate_ratio(mid2)
            
            print(f"[二分法] 比例 {mid1:.3f} 的F1分数: {score1:.4f}")
            print(f"[二分法] 比例 {mid2:.3f} 的F1分数: {score2:.4f}")
            
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
            print(f"[二分法] 当前最佳比例: {best_ratio:.3f}, F1分数: {best_score:.4f}")
            print(f"[二分法] 进度: {int((iter_count / max_iter) * 100)}%")
            
        print(f"\n[二分法] 搜索完成，最佳比例: {best_ratio:.3f}，共训练{iter_count}次")
        print(f"[最佳比例模型评估指标]")
        for k, v in best_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        return round(best_ratio, 3)

if __name__ == "__main__":
    print("\n==== 数据集比例动态调整机制（二分法） ====")
    
    try:
        print("初始化数据集优化器...")
        tuner = DynamicRatioTuner()
        print("初始化完成")
        
        print("开始二分法搜索最佳比例...")
        best_ratio = tuner.get_best_ratio()
        print(f"\n最终最佳数据集比例: {best_ratio}")
        
    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        print("请检查数据集和配置文件是否存在")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 