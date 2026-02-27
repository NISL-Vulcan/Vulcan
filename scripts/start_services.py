#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本 - 运行主后端服务（包含数据集优化功能）
"""

import os

from vulcan.services.backend_server import run_backend

def start_backend_server():
    """启动主后端服务（通过 vulcan.services.backend_server 封装）。"""
    print("🚀 启动主后端服务 (端口 5000)...")
    print("📋 包含功能:")
    print("  • 配置生成")
    print("  • 模型训练")
    print("  • 模型验证")
    print("  • 数据集优化")
    try:
        run_backend()
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
    
    # 对 src/ 布局而言，核心是包可导入与必要目录存在，而不是检查 scripts/ 薄封装脚本。
    if not os.path.exists("configs"):
        print("❌ configs 目录不存在")
        return

    try:
        import vulcan  # noqa: F401
    except Exception as e:
        print(f"❌ 无法导入 vulcan 包: {e}")
        print("请在项目根目录运行: python -m pip install -e .")
        return
    
    print("✅ 环境检查通过（src/ 安装态）")
    
    try:
        # 启动主后端服务
        start_backend_server()
        
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务...")
        print("✅ 服务已停止")

if __name__ == "__main__":
    main() 