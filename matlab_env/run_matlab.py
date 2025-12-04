#!/usr/bin/env python3
"""
Python script to execute MATLAB script with license support.
"""

import subprocess
import sys
import os
from pathlib import Path

# Ensure we're using the correct encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configuration
# Default script (can be overridden by command-line argument)
MATLAB_SCRIPT = "Matlab_SVDPCA_KNN.m"
MATLAB_PATH = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"


def load_license(license_type="MATLAB_LICENSE"):
    """
    Load license/API key from license.txt
    
    Args:
        license_type: Type of license to load (e.g., "OPENROUTER_API_KEY", "MATLAB_LICENSE")
    
    Returns:
        License value as string, or None if not found
    """
    try:
        license_path = Path("license.txt")
        if license_path.exists():
            with open(license_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Check for key=value format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == license_type:
                            return value
        return None
    except Exception as e:
        print(f"Error loading license: {e}")
        return None


# Load MATLAB license from license.txt
MATLAB_LICENSE = load_license("MATLAB_LICENSE")
if not MATLAB_LICENSE:
    print("Warning: MATLAB license not found in license.txt. Using default or no license.")
    MATLAB_LICENSE = None

# Get the directory where this script is located
# Handle case where __file__ might not be set (e.g., when run via Code Runner)
try:
    if '__file__' in globals() and __file__:
        SCRIPT_DIR = Path(__file__).parent.absolute()
    else:
        # Fallback to current working directory
        SCRIPT_DIR = Path(os.getcwd()).absolute()
except (NameError, AttributeError):
    # Fallback to current working directory
    SCRIPT_DIR = Path(os.getcwd()).absolute()

MATLAB_SCRIPT_PATH = SCRIPT_DIR / MATLAB_SCRIPT


def find_matlab():
    """Find MATLAB installation if default path doesn't exist."""
    possible_paths = [
        r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe",
        r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe",
        r"C:\Program Files\MATLAB\R2024a\bin\matlab.exe",
        r"C:\Program Files (x86)\MATLAB\R2025b\bin\matlab.exe",
        r"C:\Program Files (x86)\MATLAB\R2024b\bin\matlab.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try to find MATLAB in common locations
    import glob
    for base in [r"C:\Program Files\MATLAB", r"C:\Program Files (x86)\MATLAB"]:
        if os.path.exists(base):
            matches = glob.glob(os.path.join(base, "R*/bin/matlab.exe"))
            if matches:
                return matches[0]  # Return the first match (usually latest)
    
    return None


def execute_matlab_script(matlab_path, script_path, license_key=None):
    """
    Execute MATLAB script using MATLAB command line.
    
    Args:
        matlab_path: Path to MATLAB executable
        script_path: Path to MATLAB script (.m file)
        license_key: Optional license key
    """
    if not os.path.exists(matlab_path):
        raise FileNotFoundError(f"MATLAB not found at: {matlab_path}")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"MATLAB script not found at: {script_path}")
    
    # Change to script directory
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    
    # Build MATLAB command
    # -batch: Execute script and exit (non-interactive)
    # -nosplash: Don't show splash screen
    # -nodesktop: Don't show desktop
    # -wait: Wait for MATLAB to finish
    # -r: Run command (we'll use this to execute the script)
    
    # Use -batch mode (recommended for R2019b and later)
    # Escape single quotes in paths for MATLAB
    script_dir_escaped = script_dir.replace("'", "''")
    script_name_escaped = script_name.replace("'", "''")
    matlab_cmd = [
        matlab_path,
        "-batch",
        f"cd('{script_dir_escaped}'); run('{script_name_escaped}');"
    ]
    
    # If license is provided, you might need to set it as environment variable
    # Note: MATLAB license is typically managed by license manager, not command line
    env = os.environ.copy()
    if license_key:
        # Some MATLAB installations might use this
        env['MLM_LICENSE_FILE'] = license_key
    
    print("=" * 70)
    print(f"Executing MATLAB script: {script_name}")
    print(f"MATLAB path: {matlab_path}")
    print(f"Working directory: {script_dir}")
    if license_key:
        print(f"License: {license_key}")
    print("=" * 70)
    print()
    
    try:
        # Execute MATLAB
        result = subprocess.run(
            matlab_cmd,
            cwd=script_dir,
            env=env,
            capture_output=False,  # Show output in real-time
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        print()
        print("=" * 70)
        if result.returncode == 0:
            print("MATLAB script executed successfully!")
        else:
            print(f"MATLAB script completed with exit code: {result.returncode}")
        print("=" * 70)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error executing MATLAB: {e}")
        return False


def check_matlab_script_paths(script_path):
    """Check if paths referenced in MATLAB script exist."""
    import re
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find DATASET_PATH
        dataset_match = re.search(r"CFG\.DATASET_PATH\s*=\s*['\"]([^'\"]+)['\"]", content)
        if dataset_match:
            dataset_path = dataset_match.group(1)
            # Convert forward slashes to backslashes for Windows
            dataset_path_win = dataset_path.replace('/', '\\')
            if not os.path.exists(dataset_path_win):
                print(f"WARNING: Dataset path in MATLAB script does not exist:")
                print(f"  {dataset_path_win}")
                # Check if file exists in current directory
                filename = os.path.basename(dataset_path_win)
                if os.path.exists(filename):
                    print(f"  Found '{filename}' in current directory.")
                    print(f"  Consider updating the path in tets.m to use relative path or correct absolute path.")
                return False
        
        # Check output directory
        output_match = re.search(r"CFG\.path_saving_plot\s*=\s*['\"]([^'\"]+)['\"]", content)
        if output_match:
            output_path = output_match.group(1).replace('/', '\\')
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else output_path
            if output_dir and not os.path.exists(output_dir):
                print(f"WARNING: Output directory does not exist:")
                print(f"  {output_dir}")
                print(f"  It will be created automatically by MATLAB if possible.")
        
        return True
    except Exception as e:
        print(f"Warning: Could not check MATLAB script paths: {e}")
        return True  # Continue anyway


def main():
    """Main function."""
    # Check if script path was provided as command-line argument
    script_path_arg = None
    if len(sys.argv) > 1:
        script_path_arg = Path(sys.argv[1]).absolute()
        if script_path_arg.exists():
            script_path = script_path_arg
            print(f"Using script path from command-line argument: {script_path}")
        else:
            print(f"Warning: Provided script path does not exist: {script_path_arg}")
            print(f"Falling back to default script.")
            script_path_arg = None
    
    # Change to script's directory to ensure relative paths work
    original_cwd = os.getcwd()
    try:
        if script_path_arg:
            # Change to the directory of the provided script
            os.chdir(script_path_arg.parent)
        else:
            os.chdir(SCRIPT_DIR)
        print(f"Changed working directory to: {os.getcwd()}")
    except Exception as e:
        print(f"Warning: Could not change to script directory: {e}")
        print(f"Continuing with current directory: {os.getcwd()}")
    
    # Debug: Print current working directory and script paths
    print(f"Original working directory: {original_cwd}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {SCRIPT_DIR}")
    if script_path_arg:
        print(f"MATLAB script path (from argument): {script_path}")
    else:
        print(f"MATLAB script path (default): {MATLAB_SCRIPT_PATH}")
    print()
    
    # Determine the actual script path - try multiple locations
    if script_path_arg and script_path_arg.exists():
        script_path = script_path_arg
    else:
        script_path = MATLAB_SCRIPT_PATH
        if not script_path.exists():
            # Try current directory (should be script directory now)
            alt_path = Path(os.getcwd()) / MATLAB_SCRIPT
            if alt_path.exists():
                print(f"Warning: Script not found at {MATLAB_SCRIPT_PATH}")
                print(f"Found script at: {alt_path}")
                script_path = alt_path
            else:
                # Try relative to current directory
                print(f"Error: MATLAB script not found at: {MATLAB_SCRIPT_PATH}")
                print(f"Also checked: {alt_path}")
                print(f"Current directory contents:")
                try:
                    for f in os.listdir('.'):
                        if f.endswith('.m'):
                            print(f"  - {f}")
                except Exception as e:
                    print(f"  Could not list directory: {e}")
                # Restore original directory before exiting
                try:
                    os.chdir(original_cwd)
                except:
                    pass
                sys.exit(1)
    
    # Find MATLAB
    matlab_path = find_matlab()
    if not matlab_path:
        print("Error: Could not find MATLAB installation.")
        print("Please ensure MATLAB is installed or specify the path manually.")
        sys.exit(1)
    
    # Check paths in MATLAB script
    print("Checking MATLAB script paths...")
    check_matlab_script_paths(str(script_path))
    print()
    
    # Execute MATLAB script
    try:
        success = execute_matlab_script(
            matlab_path=matlab_path,
            script_path=str(script_path),
            license_key=MATLAB_LICENSE
        )
    finally:
        # Restore original working directory
        try:
            os.chdir(original_cwd)
        except:
            pass
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: An unexpected error occurred!")
        print("=" * 70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nDebug information:")
        print(f"  Current directory: {os.getcwd()}")
        print(f"  Script directory: {SCRIPT_DIR}")
        print(f"  Python version: {sys.version}")
        print(f"  Python executable: {sys.executable}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        sys.exit(1)

