#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查后端服务状态
"""

import requests
import sys

def check_backend():
    """检查后端服务"""
    try:
        print("🔍 检查后端服务...")
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ 后端服务正常运行")
            return True
        else:
            print("❌ 后端服务异常")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到后端服务 (ConnectionError)")
        print("请确保后端服务已启动: python start_services.py")
        return False
    except Exception as e:
        print(f"❌ 连接错误: {e}")
        return False

if __name__ == "__main__":
    if check_backend():
        print("\n✅ 后端服务正常，可以尝试数据集优化")
    else:
        print("\n❌ 后端服务有问题，请先启动后端服务")
        sys.exit(1) 