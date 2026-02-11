"""Run the StretchSense XR Glove Tracker"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).parent / "src" / "live_joint_list.py"
    subprocess.run([sys.executable, str(script)] + sys.argv[1:])
    