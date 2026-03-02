#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vulcan-Detection backend startup script.
"""

import os
import sys
import subprocess
from pathlib import Path

from vulcan.services.backend_server import run_backend

def check_dependencies():
    """Check whether required dependencies are installed."""
    try:
        import flask
        import flask_cors
        import yaml
        import requests
        import psutil
        print(" All dependencies are installed")
        return True
    except ImportError as e:
        print(f" Missing dependency: {e}")
        print("Run: pip install -r requirements_backend.txt")
        return False

def check_vulcan_environment():
    """Check vulcan runtime environment."""
    # For src layout, the key checks are package importability and config directory.
    if not Path("configs").exists():
        print(" Missing directory: configs")
        return False

    try:
        import vulcan  # noqa: F401
    except Exception as e:
        print(f" Failed to import vulcan package: {e}")
        print("Run in project root: python -m pip install -e .")
        return False

    print(" vulcan environment check passed (src install mode)")
    return True

def start_backend():
    """Start backend server via vulcan.services.backend_server."""
    print(" Starting vulcan-Detection backend server...")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check environment
    if not check_vulcan_environment():
        return False
    
    # Create required directories
    Path("generated_configs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    # Start server (actual logic is in vulcan.services.backend_server)
    try:
        run_backend()
    except KeyboardInterrupt:
        print("\n Server stopped")
    except Exception as e:
        print(f" Startup failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("="*50)
    print("vulcan-Detection Backend Server")
    print("="*50)
    
    success = start_backend()
    
    if not success:
        sys.exit(1) 