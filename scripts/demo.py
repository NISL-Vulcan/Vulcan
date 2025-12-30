#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vulcan-Detection 前端后端集成演示脚本
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

def print_header():
    """打印标题"""
    print("=" * 60)
    print("vulcan-Detection 前端后端集成演示")
    print("=" * 60)
    print()

def check_backend_status():
    """检查后端服务器状态"""
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ 后端服务器正在运行")
            return True
        else:
            print("❌ 后端服务器响应异常")
            return False
    except Exception as e:
        print(f"❌ 无法连接到后端服务器: {e}")
        print("请先启动后端服务器: python start_backend.py")
        return False

def demo_get_models():
    """演示获取可用模型"""
    print("\n🔍 获取可用模型列表...")
    try:
        response = requests.get("http://localhost:5000/api/models")
        data = response.json()
        
        if data.get("success"):
            models = data.get("models", [])
            print(f"✅ 找到 {len(models)} 个可用模型:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
        else:
            print(f"❌ 获取模型失败: {data.get('error')}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def demo_get_datasets():
    """演示获取可用数据集"""
    print("\n📊 获取可用数据集列表...")
    try:
        response = requests.get("http://localhost:5000/api/datasets")
        data = response.json()
        
        if data.get("success"):
            datasets = data.get("datasets", [])
            print(f"✅ 找到 {len(datasets)} 个可用数据集:")
            for i, dataset in enumerate(datasets, 1):
                print(f"  {i}. {dataset}")
        else:
            print(f"❌ 获取数据集失败: {data.get('error')}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def demo_generate_config():
    """演示生成配置文件"""
    print("\n🔧 生成配置文件演示...")
    
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
    
    print(f"📝 配置参数:")
    for key, value in config_request.items():
        print(f"  {key}: {value}")
    
    try:
        response = requests.post(
            "http://localhost:5000/api/generate-config",
            json=config_request
        )
        data = response.json()
        
        if data.get("success"):
            print(f"✅ {data.get('message')}")
            print(f"📁 配置文件路径: {data.get('config_path')}")
            print(f"🆔 配置ID: {data.get('config_id')}")
            
            # 显示部分配置内容
            config = data.get('config', {})
            print(f"\n📋 配置文件预览:")
            print(f"  设备: {config.get('DEVICE')}")
            print(f"  模型: {config.get('MODEL', {}).get('NAME')}")
            print(f"  数据集: {config.get('DATASET', {}).get('NAME')}")
            print(f"  训练轮数: {config.get('TRAIN', {}).get('EPOCHS')}")
            print(f"  批次大小: {config.get('TRAIN', {}).get('BATCH_SIZE')}")
            
            return data.get('config_id')
        else:
            print(f"❌ 生成配置失败: {data.get('error')}")
            return None
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def demo_start_training(config_id):
    """演示启动训练"""
    if not config_id:
        print("\n⚠️ 跳过训练演示 - 没有有效的配置ID")
        return None
    
    print("\n🚀 启动训练演示...")
    
    try:
        response = requests.post(
            "http://localhost:5000/api/start-training",
            json={"config_id": config_id}
        )
        data = response.json()
        
        if data.get("success"):
            job_id = data.get('job_id')
            print(f"✅ {data.get('message')}")
            print(f"🆔 任务ID: {job_id}")
            return job_id
        else:
            print(f"❌ 启动训练失败: {data.get('error')}")
            return None
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def demo_monitor_training(job_id):
    """演示训练监控"""
    if not job_id:
        print("\n⚠️ 跳过训练监控演示 - 没有有效的任务ID")
        return
    
    print("\n📊 训练监控演示...")
    print("正在监控训练进度（监控30秒）...")
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < 30:  # 监控30秒
        try:
            response = requests.get(f"http://localhost:5000/api/training-status/{job_id}")
            data = response.json()
            
            if data.get("success"):
                status = data.get('status')
                progress = data.get('progress', 0)
                current_epoch = data.get('current_epoch', 0)
                total_epochs = data.get('total_epochs', 0)
                metrics = data.get('metrics', {})
                
                # 只在状态变化时打印
                if status != last_status:
                    print(f"\n📈 训练状态: {status}")
                    if status == 'running':
                        print(f"⏳ 进度: {progress:.1f}%")
                        print(f"🔄 轮次: {current_epoch}/{total_epochs}")
                        if metrics.get('loss'):
                            print(f"📉 损失: {metrics['loss']:.4f}")
                        if metrics.get('accuracy'):
                            print(f"🎯 准确率: {metrics['accuracy']:.4f}")
                    last_status = status
                
                # 如果训练完成或失败，退出监控
                if status in ['completed', 'failed']:
                    print(f"\n🏁 训练{status}: 监控结束")
                    if status == 'completed':
                        print("🎉 训练成功完成！")
                    else:
                        print("💔 训练失败")
                    break
                    
            else:
                print(f"❌ 获取状态失败: {data.get('error')}")
                break
                
        except Exception as e:
            print(f"❌ 监控请求失败: {e}")
            break
        
        time.sleep(3)  # 每3秒查询一次
    
    print("\n📊 监控演示结束")

def demo_command_examples():
    """演示命令示例"""
    print("\n💬 自然语言命令示例:")
    print("以下是您可以在前端界面中使用的命令:")
    print()
    
    examples = [
        "生成配置文件，模型：DeepWuKong，数据集：DWK_Dataset，设备：cuda，epochs：20，batch_size：64",
        "生成配置文件，模型：Devign，数据集：Devign_Partial，学习率：0.0001",
        "生成配置文件，模型：IVDetect，设备：cpu，快速模式",
        "启动训练",
        "查看训练状态",
        "查看训练进度"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"  {i}. \"{example}\"")

def main():
    """主函数"""
    print_header()
    
    # 检查后端状态
    if not check_backend_status():
        return
    
    print("\n🎯 开始系统功能演示...")
    
    # 演示获取模型和数据集
    demo_get_models()
    demo_get_datasets()
    
    # 演示生成配置文件
    config_id = demo_generate_config()
    
    # 询问是否要启动训练演示
    print("\n" + "="*50)
    print("⚠️  训练演示说明:")
    print("训练演示会启动真实的训练进程，可能需要较长时间。")
    print("演示会监控训练30秒后自动结束。")
    print("="*50)
    
    response = input("\n是否要继续训练演示？(y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        # 演示启动训练
        job_id = demo_start_training(config_id)
        
        # 演示训练监控
        demo_monitor_training(job_id)
    else:
        print("⏭️  跳过训练演示")
    
    # 显示命令示例
    demo_command_examples()
    
    print("\n" + "="*60)
    print("🎉 演示完成！")
    print()
    print("接下来您可以:")
    print("1. 打开前端界面进行可视化操作")
    print("2. 使用API接口进行程序化调用")
    print("3. 查看生成的配置文件: generated_configs/")
    print("="*60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        sys.exit(1) 