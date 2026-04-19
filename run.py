"""
Convenience startup script for Mazag backend.
Run with: python run.py
Or directly with: uvicorn api.main:app --reload --port 8000
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # Must be run from mazag-backend/ directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--reload-dir", "api",
    ]
    print("Starting Mazag backend on http://localhost:8000 ...")
    subprocess.run(cmd)
