#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vulcan-Detection 后端启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import flask
        import flask_cors
        import yaml
        import requests
        import psutil
        print("✅ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements_backend.txt")
        return False

def check_vulcan_environment():
    """检查vulcan环境"""
    # 检查主要目录
    required_dirs = ['configs', 'framework', 'tools']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"❌ 缺少目录: {dir_name}")
            return False
    
    # 检查主要文件
    required_files = ['tools/train.py']
    for file_name in required_files:
        if not Path(file_name).exists():
            print(f"❌ 缺少文件: {file_name}")
            return False
    
    print("✅ vulcan环境检查通过")
    return True

def start_backend():
    """启动后端服务器"""
    print("🚀 启动vulcan-Detection后端服务器...")
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    # 检查环境
    if not check_vulcan_environment():
        return False
    
    # 创建必要的目录
    Path("generated_configs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    # 启动服务器
    try:
        subprocess.run([sys.executable, "backend_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("="*50)
    print("vulcan-Detection 后端服务器")
    print("="*50)
    
    success = start_backend()
    
    if not success:
        sys.exit(1) 