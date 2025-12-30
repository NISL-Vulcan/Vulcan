#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本
"""

import os
import sys
import subprocess
import time

def quick_test():
    print("🔍 快速诊断数据集优化问题")
    print("=" * 40)
    
    # 1. 检查文件
    script_path = "auto_update_and_dynamic_ratio.py"
    if not os.path.exists(script_path):
        print("❌ 优化脚本不存在")
        return
    
    print("✅ 优化脚本存在")
    
    # 2. 尝试运行脚本（最多10秒）
    print("\n🚀 尝试运行脚本...")
    
    try:
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        
        print(f"进程PID: {process.pid}")
        
        # 读取输出10秒
        start_time = time.time()
        output = []
        
        while time.time() - start_time < 10:
            line = process.stdout.readline()
            if line:
                line = line.rstrip()
                print(f"📝 {line}")
                output.append(line)
            else:
                time.sleep(0.1)
        
        # 终止进程
        process.terminate()
        process.wait(timeout=3)
        
        print(f"\n📊 结果:")
        print(f"输出行数: {len(output)}")
        print(f"返回码: {process.returncode}")
        
        if len(output) == 0:
            print("❌ 没有输出，脚本可能卡住了")
        else:
            print("✅ 脚本有输出，工作正常")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    quick_test() 