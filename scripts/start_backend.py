#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vulcan-Detection 后端启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

from vulcan.services.backend_server import run_backend

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
    # 对 src/ 布局而言，关键是“包可导入”与配置目录存在，而不是依赖 tools/ 目录。
    if not Path("configs").exists():
        print("❌ 缺少目录: configs")
        return False

    try:
        import vulcan  # noqa: F401
    except Exception as e:
        print(f"❌ 无法导入 vulcan 包: {e}")
        print("请在项目根目录运行: python -m pip install -e .")
        return False

    print("✅ vulcan环境检查通过（src/ 安装态）")
    return True

def start_backend():
    """启动后端服务器（通过 vulcan.services.backend_server 封装）。"""
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
    
    # 启动服务器（实际逻辑在 vulcan.services.backend_server 中）
    try:
        run_backend()
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