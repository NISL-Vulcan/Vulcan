#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check backend service status.
"""

import requests
import sys

def check_backend():
    """Check backend service."""
    try:
        print(" Checking backend service...")
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(" Backend service is running")
            return True
        else:
            print(" Backend service returned an unhealthy response")
            return False
    except requests.exceptions.ConnectionError:
        print(" Cannot connect to backend service (ConnectionError)")
        print("Please start backend first: python start_services.py")
        return False
    except Exception as e:
        print(f" Connection error: {e}")
        return False

if __name__ == "__main__":
    if check_backend():
        print("\n Backend is healthy, you can try dataset optimization")
    else:
        print("\n Backend has issues, please start it first")
        sys.exit(1) 