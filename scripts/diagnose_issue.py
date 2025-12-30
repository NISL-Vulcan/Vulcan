#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整诊断数据集优化问题
"""

import os
import sys
import requests
import subprocess
import time
import json

def check_backend_service():
    """检查后端服务"""
    print("🔍 1. 检查后端服务")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ 后端服务正常运行")
            return True
        else:
            print(f"❌ 后端服务异常，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到后端服务: {e}")
        return False

def check_files():
    """检查必要文件"""
    print("\n🔍 2. 检查必要文件")
    print("-" * 30)
    
    files_to_check = [
        "auto_update_and_dynamic_ratio.py",
        "backend_server.py",
        "dataset/vulnerables.jsonl",
        "dataset/non-vulnerables.jsonl",
        "configs/regvd_reveal.yaml"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 不存在")
            all_exist = False
    
    return all_exist

def test_optimization_script():
    """测试优化脚本"""
    print("\n🔍 3. 测试优化脚本")
    print("-" * 30)
    
    try:
        # 尝试导入脚本
        import auto_update_and_dynamic_ratio
        print("✅ 脚本导入成功")
        
        # 尝试创建实例
        tuner = auto_update_and_dynamic_ratio.DynamicRatioTuner()
        print("✅ 实例创建成功")
        
        return True
    except Exception as e:
        print(f"❌ 脚本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """测试API端点"""
    print("\n🔍 4. 测试API端点")
    print("-" * 30)
    
    try:
        # 测试启动优化API
        print("测试启动优化API...")
        response = requests.post(
            "http://localhost:5000/api/start-dataset-optimization",
            json={"max_iterations": 1},  # 只测试1次迭代
            timeout=10
        )
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if result.get("success"):
                job_id = result["job_id"]
                print(f"✅ 优化任务启动成功，任务ID: {job_id}")
                
                # 测试状态查询
                print("\n测试状态查询API...")
                status_response = requests.get(f"http://localhost:5000/api/optimization-status/{job_id}", timeout=5)
                
                if status_response.status_code == 200:
                    status_result = status_response.json()
                    print(f"状态查询成功: {json.dumps(status_result, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    print(f"❌ 状态查询失败，状态码: {status_response.status_code}")
                    return False
            else:
                print(f"❌ 启动优化失败: {result.get('error')}")
                return False
        else:
            print(f"❌ 启动优化API失败，状态码: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

def main():
    """主诊断函数"""
    print("🚀 开始诊断数据集优化问题")
    print("=" * 50)
    
    # 1. 检查后端服务
    backend_ok = check_backend_service()
    
    # 2. 检查文件
    files_ok = check_files()
    
    # 3. 测试优化脚本
    script_ok = test_optimization_script()
    
    # 4. 测试API端点
    api_ok = False
    if backend_ok and files_ok and script_ok:
        api_ok = test_api_endpoints()
    
    # 总结
    print("\n📋 诊断结果")
    print("=" * 50)
    print(f"后端服务: {'✅ 正常' if backend_ok else '❌ 异常'}")
    print(f"必要文件: {'✅ 完整' if files_ok else '❌ 缺失'}")
    print(f"优化脚本: {'✅ 正常' if script_ok else '❌ 异常'}")
    print(f"API端点: {'✅ 正常' if api_ok else '❌ 异常'}")
    
    if backend_ok and files_ok and script_ok and api_ok:
        print("\n✅ 所有检查都通过，数据集优化功能应该正常工作")
        print("如果前端仍然卡住，可能是前端代码问题")
    else:
        print("\n❌ 发现问题，需要修复")
        if not backend_ok:
            print("- 请启动后端服务: python start_services.py")
        if not files_ok:
            print("- 请检查必要文件是否存在")
        if not script_ok:
            print("- 请检查优化脚本是否有错误")
        if not api_ok:
            print("- 请检查API端点配置")

if __name__ == "__main__":
    main() 