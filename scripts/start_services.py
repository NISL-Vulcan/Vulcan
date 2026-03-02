#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Startup script - run main backend service (including dataset optimization).
"""

import os

from vulcan.services.backend_server import run_backend

def start_backend_server():
    """Start main backend service via vulcan.services.backend_server."""
    print(" Starting main backend service (port 5000)...")
    print(" Included features:")
    print("  • Config generation")
    print("  • Model training")
    print("  • Model validation")
    print("  • Dataset optimization")
    try:
        run_backend()
    except KeyboardInterrupt:
        print("\n Main backend service stopped")
    except Exception as e:
        print(f" Main backend startup failed: {e}")

def main():
    print("=" * 60)
    print(" vulcan-Detection Service Launcher")
    print("=" * 60)
    print(" The following services will be started:")
    print("  • Main backend service (port 5000) - all features included")
    print("  • Frontend app (port 5173) - start manually")
    print("=" * 60)
    
    # For src layout, key checks are package importability and required directories.
    if not os.path.exists("configs"):
        print(" configs directory does not exist")
        return

    try:
        import vulcan  # noqa: F401
    except Exception as e:
        print(f" Failed to import vulcan package: {e}")
        print("Run in project root: python -m pip install -e .")
        return
    
    print(" Environment check passed (src install mode)")
    
    try:
        # Start main backend
        start_backend_server()
        
    except KeyboardInterrupt:
        print("\n Stopping services...")
        print(" Services stopped")

if __name__ == "__main__":
    main() 