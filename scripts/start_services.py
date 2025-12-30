#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本 - 运行主后端服务（包含数据集优化功能）
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def start_backend_server():
    """启动主后端服务"""
    print("🚀 启动主后端服务 (端口 5000)...")
    print("📋 包含功能:")
    print("  • 配置生成")
    print("  • 模型训练")
    print("  • 模型验证")
    print("  • 数据集优化")
    try:
        subprocess.run([sys.executable, "backend_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 主后端服务已停止")
    except Exception as e:
        print(f"❌ 主后端服务启动失败: {e}")

def main():
    print("=" * 60)
    print("🚀 vulcan-Detection 服务启动器")
    print("=" * 60)
    print("📋 将启动以下服务:")
    print("  • 主后端服务 (端口 5000) - 包含所有功能")
    print("  • 前端应用 (端口 5173) - 需要手动启动")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists("backend_server.py"):
        print("❌ backend_server.py 不存在")
        return
    
    if not os.path.exists("auto_update_and_dynamic_ratio.py"):
        print("❌ auto_update_and_dynamic_ratio.py 不存在")
        return
    
    print("✅ 所有必需文件已找到")
    
    try:
        # 启动主后端服务
        start_backend_server()
        
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务...")
        print("✅ 服务已停止")

if __name__ == "__main__":
    main() 