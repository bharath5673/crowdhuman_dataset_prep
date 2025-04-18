import os
import subprocess

# === Config ===
target_dir = "yolov5-crowdhuman"
script_name = "detect2.py"

# Change to the target directory
os.chdir(target_dir)

# Run the script
subprocess.run(["python3", script_name])
