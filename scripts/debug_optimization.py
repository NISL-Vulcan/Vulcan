#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试数据集优化功能
"""

import os
import sys
import subprocess

def check_environment():
    """检查环境"""
    print("🔍 检查环境...")
    
    # 检查Python版本
    print(f"🐍 Python版本: {sys.version}")
    print(f"🐍 Python路径: {sys.executable}")
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"📁 当前目录: {current_dir}")
    
    # 检查必需文件
    required_files = [
        "backend_server.py",
        "auto_update_and_dynamic_ratio.py"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} 存在")
        else:
            print(f"❌ {file} 不存在")
    
    # 检查数据集目录
    dataset_dir = "dataset"
    if os.path.exists(dataset_dir):
        print(f"✅ 数据集目录存在: {dataset_dir}")
        dataset_files = os.listdir(dataset_dir)
        print(f"📊 数据集文件: {dataset_files}")
    else:
        print(f"❌ 数据集目录不存在: {dataset_dir}")
    
    # 检查配置文件目录
    configs_dir = "configs"
    if os.path.exists(configs_dir):
        print(f"✅ 配置文件目录存在: {configs_dir}")
        config_files = [f for f in os.listdir(configs_dir) if f.endswith('.yaml')]
        print(f"📄 配置文件: {config_files}")
    else:
        print(f"❌ 配置文件目录不存在: {configs_dir}")

def test_optimization_script():
    """测试优化脚本"""
    print("\n🧪 测试优化脚本...")
    
    try:
        # 直接运行优化脚本
        cmd = [sys.executable, "auto_update_and_dynamic_ratio.py"]
        print(f"🚀 运行命令: {' '.join(cmd)}")
        
        # 运行脚本并捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ, PYTHONUNBUFFERED='1')
        )
        
        print("📝 脚本输出:")
        print("-" * 50)
        
        # 读取前几行输出
        line_count = 0
        for line in process.stdout:
            print(line.rstrip())
            line_count += 1
            if line_count >= 20:  # 只显示前20行
                print("... (输出被截断)")
                break
        
        # 等待进程完成
        return_code = process.wait()
        print("-" * 50)
        print(f"📊 脚本返回码: {return_code}")
        
        if return_code == 0:
            print("✅ 优化脚本运行成功")
        else:
            print("❌ 优化脚本运行失败")
            
    except Exception as e:
        print(f"❌ 运行优化脚本异常: {e}")

def main():
    """主函数"""
    print("🚀 调试数据集优化功能")
    print("=" * 60)
    
    # 检查环境
    check_environment()
    
    # 测试优化脚本
    test_optimization_script()
    
    print("\n📋 调试完成")
    print("\n💡 建议:")
    print("1. 如果优化脚本运行成功，说明环境配置正确")
    print("2. 如果脚本运行失败，请检查错误信息")
    print("3. 确保数据集文件存在且格式正确")
    print("4. 确保配置文件存在且格式正确")

if __name__ == "__main__":
    main() 