#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to run training scripts with custom dataset path
Usage: python run_training_wrapper.py <script_name> <dataset_path>
"""

import sys
import os
import re
from pathlib import Path

# Set UTF-8 encoding for stdout/stderr
import locale
try:
    if sys.platform != 'win32':
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    else:
        # On Windows, try to set UTF-8
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
except locale.Error:
    # If locale setting fails, continue anyway
    pass

# Ensure UTF-8 encoding for output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Set environment variables for UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'

if len(sys.argv) < 3:
    print("Usage: python run_training_wrapper.py <script_name> <dataset_path>")
    sys.exit(1)

script_name = sys.argv[1]
dataset_path = sys.argv[2]

# Add universal_training_script to path
script_dir = Path("/app/universal_training_script")
sys.path.insert(0, str(script_dir))

# Read the script
script_path = script_dir / script_name
if not script_path.exists():
    print(f"Error: Script '{script_name}' not found")
    sys.exit(1)

# Read script content with UTF-8 encoding
try:
    with open(script_path, 'r', encoding='utf-8', errors='replace') as f:
        script_content = f.read()
except UnicodeDecodeError as e:
    print(f"Error reading script file: {e}", file=sys.stderr)
    sys.exit(1)

# Replace DATASET_PATH in CFG class
# Handle different variations of DATASET_PATH assignment
# First, use regex to replace any DATASET_PATH assignment with the provided dataset_path
# This handles all variations of spacing and path formats
pattern = r'(DATASET_PATH\s*=\s*["\'])([^"\']+)(["\'])'
script_content = re.sub(pattern, f'\\g<1>{dataset_path}\\g<3>', script_content)

# Also handle specific common paths as fallback
replacements = [
    ('DATASET_PATH   = "/content/test.csv"', f'DATASET_PATH   = "{dataset_path}"'),
    ('DATASET_PATH     = "/content/test.csv"', f'DATASET_PATH     = "{dataset_path}"'),
    ('DATASET_PATH = "/content/test.csv"', f'DATASET_PATH = "{dataset_path}"'),
    ('DATASET_PATH   = "/content/output.csv"', f'DATASET_PATH   = "{dataset_path}"'),
    ('DATASET_PATH     = "/content/output.csv"', f'DATASET_PATH     = "{dataset_path}"'),
    ('DATASET_PATH = "/content/output.csv"', f'DATASET_PATH = "{dataset_path}"'),
    ('DATASET_PATH   = "/content/dataset.csv"', f'DATASET_PATH   = "{dataset_path}"'),
    ('DATASET_PATH     = "/content/dataset.csv"', f'DATASET_PATH     = "{dataset_path}"'),
    ('DATASET_PATH = "/content/dataset.csv"', f'DATASET_PATH = "{dataset_path}"'),
    ('DATASET_PATH   = "/app/upload/dataset.csv"', f'DATASET_PATH   = "{dataset_path}"'),
    ('DATASET_PATH     = "/app/upload/dataset.csv"', f'DATASET_PATH     = "{dataset_path}"'),
    ('DATASET_PATH = "/app/upload/dataset.csv"', f'DATASET_PATH = "{dataset_path}"'),
]

for old, new in replacements:
    script_content = script_content.replace(old, new)

# Update path_saving_plot and performance_plot_name, then PLOT_SAVE_PATH
# Handle different variations of path_saving_plot assignment
plot_path_replacements = [
    ('path_saving_plot = "/content/output"', 'path_saving_plot = "/app/output"'),
    ('path_saving_plot   = "/content/output"', 'path_saving_plot   = "/app/output"'),
    ('path_saving_plot     = "/content/output"', 'path_saving_plot     = "/app/output"'),
    ('path_saving_plot = "/content"', 'path_saving_plot = "/app/output"'),
    ('path_saving_plot   = "/content"', 'path_saving_plot   = "/app/output"'),
    ('path_saving_plot     = "/content"', 'path_saving_plot     = "/app/output"'),
]

for old, new in plot_path_replacements:
    script_content = script_content.replace(old, new)

# Ensure path_saving_plot exists (add if missing)
if 'path_saving_plot' not in script_content:
    # Find where PLOT_SAVE_PATH is defined and add path_saving_plot before it
    script_content = script_content.replace(
        'PLOT_SAVE_PATH',
        'path_saving_plot = "/app/output"  # Directory for saving plots\n    PLOT_SAVE_PATH',
        1  # Only replace first occurrence
    )

# Ensure performance_plot_name exists (add if missing)
if 'performance_plot_name' not in script_content:
    # Find where PLOT_SAVE_PATH is defined and add performance_plot_name before it
    script_content = script_content.replace(
        'PLOT_SAVE_PATH',
        'performance_plot_name = "performance_summary.png"  # Name of performance plot\n    PLOT_SAVE_PATH',
        1  # Only replace first occurrence
    )

# Update PLOT_SAVE_PATH to use os.path.join (for backward compatibility with old scripts)
# Only update if it's still using a simple string path
if 'PLOT_SAVE_PATH = "performance_summary.png"' in script_content:
    script_content = script_content.replace(
        'PLOT_SAVE_PATH = "performance_summary.png"',
        'PLOT_SAVE_PATH = os.path.join(path_saving_plot, performance_plot_name)  # Full path'
    )
if 'PLOT_SAVE_PATH   = "performance_summary.png"' in script_content:
    script_content = script_content.replace(
        'PLOT_SAVE_PATH   = "performance_summary.png"',
        'PLOT_SAVE_PATH   = os.path.join(path_saving_plot, performance_plot_name)  # Full path'
    )
if 'PLOT_SAVE_PATH     = "performance_summary.png"' in script_content:
    script_content = script_content.replace(
        'PLOT_SAVE_PATH     = "performance_summary.png"',
        'PLOT_SAVE_PATH     = os.path.join(path_saving_plot, performance_plot_name)  # Full path'
    )

# Execute the modified script with proper encoding handling
try:
    # Compile with UTF-8 source encoding
    code = compile(script_content, str(script_path), 'exec', dont_inherit=True)
    exec(code, {'__name__': '__main__', '__file__': str(script_path)})
except Exception as e:
    # Print error with proper encoding
    error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
    print(f"Error executing script: {error_msg}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

