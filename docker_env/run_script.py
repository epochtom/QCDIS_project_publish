#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python script to execute any training script in Docker
Usage: python run_script.py <script_name> [additional_args]
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_script.py <script_name> [additional_args]")
        print("Example: python run_script.py QPCA_Regression.py")
        print("Example: python run_script.py QPCA_CNN.py --arg1 value1")
        sys.exit(1)
    
    script_name = sys.argv[1]
    additional_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Get paths
    docker_env_dir = Path(__file__).parent
    project_root = docker_env_dir.parent
    script_path = project_root / "universal_training_script" / script_name
    
    # Check if script exists
    if not script_path.exists():
        print(f"Error: Script '{script_name}' not found in universal_training_script folder")
        sys.exit(1)
    
    # Build Docker image if not exists
    print("Checking Docker image...")
    result = subprocess.run(
        ["docker", "images", "universal-training:latest", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True
    )
    
    if not result.stdout.strip():
        print("Building Docker image...")
        dockerfile_path = docker_env_dir / "Dockerfile"
        build_result = subprocess.run(
            ["docker", "build", "-t", "universal-training:latest", "-f", str(dockerfile_path), str(project_root)],
            cwd=project_root
        )
        if build_result.returncode != 0:
            print("Error: Failed to build Docker image")
            sys.exit(1)
    
    # Prepare paths
    upload_path = project_root / "upload"
    output_path = project_root / "output"
    script_dir = project_root / "universal_training_script"
    
    # Create directories if they don't exist
    upload_path.mkdir(exist_ok=True)
    output_path.mkdir(exist_ok=True)
    
    # Run the script in Docker with 24-hour timeout
    print(f"Executing {script_name} in Docker container...")
    print("Timeout: 24 hours")
    print("")
    
    docker_cmd = [
        "docker", "run", "--rm",
        "--name", "universal-training-run",
        "--stop-timeout", "86400",  # 24 hours in seconds
        "-v", f"{upload_path}:/app/data:ro",
        "-v", f"{output_path}:/app/output:rw",
        "-v", f"{script_dir}:/app/universal_training_script:ro",
        "-w", "/app",
        "universal-training:latest",
        "python", f"universal_training_script/{script_name}"
    ]
    
    docker_cmd.extend(additional_args)
    
    try:
        result = subprocess.run(docker_cmd, check=True)
        print("")
        print("Execution completed successfully!")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print("")
        print(f"Execution failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("")
        print("Execution interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()

