#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单调试数据集优化脚本
"""

import os
import sys
import subprocess
import time

def test_optimization_script_direct():
    """直接测试优化脚本"""
    print("🔍 直接测试优化脚本")
    print("=" * 50)
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前目录: {current_dir}")
    
    # 检查优化脚本是否存在
    script_path = os.path.join(current_dir, "auto_update_and_dynamic_ratio.py")
    print(f"脚本路径: {script_path}")
    
    if not os.path.exists(script_path):
        print("❌ 优化脚本不存在！")
        return False
    
    print("✅ 优化脚本存在")
    
    # 检查数据集目录
    dataset_dir = os.path.join(current_dir, "dataset")
    if os.path.exists(dataset_dir):
        print(f"✅ 数据集目录存在: {dataset_dir}")
        files = os.listdir(dataset_dir)
        print(f"   文件列表: {files}")
    else:
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return False
    
    # 检查configs目录
    configs_dir = os.path.join(current_dir, "configs")
    if os.path.exists(configs_dir):
        print(f"✅ 配置目录存在: {configs_dir}")
        files = os.listdir(configs_dir)
        print(f"   文件列表: {files}")
    else:
        print(f"❌ 配置目录不存在: {configs_dir}")
        return False
    
    # 尝试直接运行脚本（只运行几秒钟）
    print("\n🚀 尝试运行优化脚本（5秒后停止）...")
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # 启动进程
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=current_dir,
            env=env
        )
        
        print(f"✅ 进程已启动，PID: {process.pid}")
        
        # 读取输出5秒
        start_time = time.time()
        output_lines = []
        
        while time.time() - start_time < 5:
            line = process.stdout.readline()
            if line:
                line = line.rstrip()
                print(f"📝 {line}")
                output_lines.append(line)
            else:
                time.sleep(0.1)
        
        # 终止进程
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        
        print(f"\n📊 输出统计:")
        print(f"   • 总输出行数: {len(output_lines)}")
        print(f"   • 进程返回码: {process.returncode}")
        
        if output_lines:
            print(f"   • 前5行输出:")
            for i, line in enumerate(output_lines[:5]):
                print(f"     {i+1}: {line}")
        
        return len(output_lines) > 0
        
    except Exception as e:
        print(f"❌ 运行脚本时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_script_import():
    """测试导入优化脚本"""
    print("\n🔍 测试导入优化脚本")
    print("=" * 50)
    
    try:
        # 添加当前目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # 尝试导入
        print("尝试导入 auto_update_and_dynamic_ratio...")
        import auto_update_and_dynamic_ratio
        
        print("✅ 导入成功")
        
        # 尝试创建实例
        print("尝试创建 DynamicRatioTuner 实例...")
        tuner = auto_update_and_dynamic_ratio.DynamicRatioTuner()
        print("✅ 实例创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始调试数据集优化脚本...")
    
    # 测试1: 直接运行
    direct_test = test_optimization_script_direct()
    
    # 测试2: 导入测试
    import_test = test_optimization_script_import()
    
    print("\n📋 测试结果:")
    print(f"直接运行测试: {'✅ 通过' if direct_test else '❌ 失败'}")
    print(f"导入测试: {'✅ 通过' if import_test else '❌ 失败'}")
    
    if direct_test and import_test:
        print("\n✅ 优化脚本工作正常")
    else:
        print("\n❌ 优化脚本有问题，需要修复") 