#!/usr/bin/env python3
# Simple launcher for IAM Access Prediction Demo

import subprocess
import sys
import os

def main():
    print("IAM Access Prediction Demo Launcher")
    print("=" * 50)
    
    if not os.path.exists("main.py"):
        print("main.py not found. Please run from the project root directory.")
        return
    
    if not os.path.exists("streamlit_modules"):
        print("streamlit_modules directory not found.")
        return
    
    print("Launching Streamlit demo...")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch Streamlit: {e}")
    except KeyboardInterrupt:
        print("\nDemo stopped by user")

if __name__ == "__main__":
    main()
