# -*- coding: utf-8 -*-
"""
Show Script and Execute GUI
Displays the generated training script and allows execution in Docker.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
import threading

try:
    from LLM_integration import create_llm_chat_widget
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM integration not available")

try:
    from log_history_helper_functions.version_history import VersionHistoryManager
    VERSION_HISTORY_AVAILABLE = True
except ImportError:
    VERSION_HISTORY_AVAILABLE = False
    print("Warning: Version history not available")

try:
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


def create_script_editor_popup(root_window, shared_state, status_text_callback, sanitize_path_for_display, sanitize_line_for_display):
    """Create a popup window with script editor and Start Training button
    
    Args:
        root_window: Root window for the popup
        shared_state: Shared state object to get methodology and script info
        status_text_callback: Callback function to update status in main window
        sanitize_path_for_display: Function to sanitize paths
        sanitize_line_for_display: Function to sanitize lines
    """
    # Initialize version history manager
    version_manager = None
    if VERSION_HISTORY_AVAILABLE:
        version_manager = VersionHistoryManager()
    
    popup = tk.Toplevel(root_window)
    popup.title("Python Editor")
    popup.geometry("900x700")
    
    # Main frame
    main_frame = ttk.Frame(popup, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="Python Editor", font=("Arial", 16, "bold"))
    title_label.grid(row=0, column=0, pady=(0, 10))
    
    # Methodology info label
    info_label_frame = ttk.Frame(main_frame)
    info_label_frame.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
    
    # Script display area
    script_frame = ttk.LabelFrame(main_frame, text="Training Script (Editable)", padding="10")
    script_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    script_frame.columnconfigure(0, weight=1)
    script_frame.rowconfigure(1, weight=1)
    main_frame.rowconfigure(2, weight=1)
    
    # Version counter for script modifications
    # Initialize by loading latest version from metadata if available
    script_version = 1
    if version_manager and shared_state and shared_state.script_name:
        latest_version = version_manager.get_latest_version(shared_state.script_name)
        if latest_version is not None:
            script_version = latest_version
            print(f"DEBUG: Loaded latest version {script_version} from metadata for {shared_state.script_name}")
    # Store version in popup for access in execution threads
    popup.script_version = script_version
    
    # Function to update script frame title with version number and environment
    def update_script_frame_title():
        """Update the script frame title with current version number and environment"""
        environment_text = ""
        if shared_state and shared_state.script_name:
            if shared_state.script_name.endswith('.m'):
                environment_text = "\nmatlab environment connected"
            else:
                environment_text = "\npython environment connected"
        
        script_frame.config(text=f"Training Script (Editable) - version {script_version}{environment_text}")
    
    # Initialize title with version
    update_script_frame_title()
    
    # Function to increment version
    def increment_version():
        """Increment script version number and update title"""
        nonlocal script_version
        script_version += 1
        popup.script_version = script_version  # Update stored version
        print(f"DEBUG: Incrementing version to {script_version}")
        update_script_frame_title()
        print(f"DEBUG: Version title updated to: {script_frame.cget('text')}")
    
    # Save button for script edits
    def save_script():
        """Save edited script to temporary folder"""
        if not shared_state or not shared_state.script_name:
            messagebox.showwarning("Warning", "No script selected to save.")
            return
        
        script_path = Path("temporary_model_training_script") / shared_state.script_name
        try:
            script_content = script_text.get(1.0, tk.END)
            
            # Save version history before saving new version
            if version_manager:
                # Get current content to save as previous version
                if script_path.exists():
                    with open(script_path, 'r', encoding='utf-8') as f:
                        old_content = f.read()
                    # Save old version
                    version_manager.save_version(shared_state.script_name, old_content, "manual", script_version)
            
            # Save new version
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            increment_version()  # Increment version when script is saved
            
            # Save new version to history
            if version_manager:
                version_manager.save_version(shared_state.script_name, script_content, "manual", script_version)
            
            messagebox.showinfo("Success", f"Script saved to {script_path}")
            if status_text_callback:
                status_text_callback(f"✓ Script saved: {script_path} (v{script_version})\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save script: {str(e)}")
            if status_text_callback:
                status_text_callback(f"✗ Error saving script: {str(e)}\n")
    
    def download_script():
        """Download script to user-selected location"""
        if not shared_state or not shared_state.script_name:
            messagebox.showwarning("Warning", "No script selected to download.")
            return
        
        try:
            script_content = script_text.get(1.0, tk.END)
            
            # Get file extension from script name
            script_name = shared_state.script_name
            file_ext = Path(script_name).suffix
            
            # Open file dialog to select save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=file_ext,
                filetypes=[("Script files", f"*{file_ext}"), ("All files", "*.*")],
                initialfile=script_name,
                title="Save Script As"
            )
            
            if file_path:  # User didn't cancel
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(script_content)
                messagebox.showinfo("Success", f"Script downloaded to {file_path}")
                if status_text_callback:
                    status_text_callback(f"✓ Script downloaded: {file_path}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download script: {str(e)}")
            if status_text_callback:
                status_text_callback(f"✗ Error downloading script: {str(e)}\n")
    
    # Button frame for Save and Download buttons
    button_row_frame = ttk.Frame(script_frame)
    button_row_frame.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
    
    save_btn = ttk.Button(button_row_frame, text="💾 Save Script", command=save_script)
    save_btn.pack(side=tk.LEFT, padx=(0, 5))
    
    download_btn = ttk.Button(button_row_frame, text="⬇️ Download", command=download_script)
    download_btn.pack(side=tk.LEFT)
    
    script_text = scrolledtext.ScrolledText(script_frame, height=30, width=80, wrap=tk.NONE, font=("Consolas", 10))
    script_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Function to map methodology to script name
    def map_methodology_to_script(methodology):
        """Map methodology string to script filename"""
        if not methodology:
            return None
        methodology_lower = methodology.lower()
        if "matlab" in methodology_lower and "svdpca" in methodology_lower:
            if "knn" in methodology_lower:
                return "Matlab_SVDPCA_KNN.m"
        elif "pqk" in methodology_lower:
            if "nn" in methodology_lower or "neural network" in methodology_lower:
                return "Python_PQK_NN.py"
        elif "qpca" in methodology_lower or "quantum pca" in methodology_lower:
            if "xgboost" in methodology_lower:
                return "Python_QPCA_XGBoost.py"
            elif "cnn" in methodology_lower:
                return "Python_QPCA_CNN.py"
            elif "regression" in methodology_lower or "linear" in methodology_lower:
                return "Python_QPCA_Regression.py"
        elif "qfm" in methodology_lower or "quantum feature mapping" in methodology_lower:
            if "xgboost" in methodology_lower:
                return "Python_QFM_XGBoost.py"
            elif "cnn" in methodology_lower:
                return "Python_QFM_CNN.py"
            elif "regression" in methodology_lower or "linear" in methodology_lower:
                return "Python_QFM_Regression.py"
        return None
    
    # Track last loaded script name to detect new script loads
    last_loaded_script = None
    
    # Function to update script content
    def update_script_content():
        """Update script display based on shared state"""
        nonlocal script_version, last_loaded_script
        
        # Clear existing info label if any
        for widget in info_label_frame.winfo_children():
            widget.destroy()
        
        # Get methodology and script name from shared state
        if shared_state and shared_state.methodology:
            methodology = shared_state.methodology
            script_name = map_methodology_to_script(methodology)
            
            if script_name:
                # Show methodology info
                info_label = ttk.Label(
                    info_label_frame, 
                    text=f"Methodology: {methodology}",
                    font=("Arial", 10, "italic"),
                    foreground="blue"
                )
                info_label.pack(side=tk.LEFT)
                
                # Store script name in shared state for execution
                shared_state.script_name = script_name
                
                # Update button text based on script type
                if script_name.endswith('.m'):
                    train_btn.config(text="🤖 Start Training (MATLAB)")
                else:
                    train_btn.config(text="🤖 Start Training (Python)")
                
                # Only reset version to 1 when loading a NEW script (different script name)
                # This prevents resetting version after LLM modifications (which keep same script)
                is_new_script = (script_name != last_loaded_script)
                if is_new_script:
                    # New script detected - check if versions exist in metadata
                    if version_manager:
                        latest_version = version_manager.get_latest_version(script_name)
                        if latest_version is not None:
                            # Script has version history - use latest version
                            script_version = latest_version
                            popup.script_version = script_version
                            print(f"DEBUG: New script loaded, using latest version {script_version} from metadata: {script_name}")
                        else:
                            # No version history - start at version 1
                            script_version = 1
                            popup.script_version = script_version
                            print(f"DEBUG: New script loaded, no version history, starting at version 1: {script_name}")
                    else:
                        # No version manager - start at version 1
                        script_version = 1
                        popup.script_version = script_version
                        print(f"DEBUG: New script loaded, no version manager, starting at version 1: {script_name}")
                    last_loaded_script = script_name
                else:
                    # Same script - load latest version from metadata to ensure we're in sync
                    if version_manager:
                        latest_version = version_manager.get_latest_version(script_name)
                        if latest_version is not None and latest_version > script_version:
                            # Metadata has a newer version - update to match
                            script_version = latest_version
                            popup.script_version = script_version
                            print(f"DEBUG: Same script detected, updated version to {script_version} from metadata")
                        else:
                            print(f"DEBUG: Refreshing same script, preserving version: {script_version}")
                    else:
                        print(f"DEBUG: Refreshing same script, preserving version: {script_version}")
                
                update_script_frame_title()  # Update title with environment info
                
                # Update script display (read from temporary folder)
                script_text.config(state=tk.NORMAL)
                script_text.delete(1.0, tk.END)
                script_path = Path("temporary_model_training_script") / script_name
                if script_path.exists():
                    # Save version 1 when script is first loaded (only for new scripts)
                    if is_new_script and version_manager:
                        try:
                            existing_versions = version_manager.get_all_versions(script_name)
                            version_1_exists = any(v["version"] == 1 for v in existing_versions)
                            if not version_1_exists:
                                with open(script_path, 'r', encoding='utf-8') as f:
                                    initial_content = f.read()
                                version_manager.save_version(script_name, initial_content, "initial_copy", 1)
                                print(f"DEBUG: Saved initial version 1 for {script_name}")
                        except Exception as e:
                            print(f"Warning: Could not save initial version 1: {e}")
                    
                    # Now read and display the script
                    try:
                        with open(script_path, 'r', encoding='utf-8') as f:
                            script_content = f.read()
                        script_text.insert(1.0, script_content)
                        script_text.config(state=tk.NORMAL)  # Allow editing in temporary folder
                    except Exception as e:
                        script_text.insert(1.0, f"Error loading script: {str(e)}")
                        script_text.config(state=tk.DISABLED)
                else:
                    script_text.insert(1.0, f"Script file not found: {script_path}\n\nPlease go back and click 'Start Generate Model Training Script' first.")
                    script_text.config(state=tk.DISABLED)
            else:
                error_label = ttk.Label(
                    info_label_frame, 
                    text=f"Could not map methodology '{methodology}' to a script.",
                    foreground="red"
                )
                error_label.pack(side=tk.LEFT)
                script_text.config(state=tk.NORMAL)
                script_text.delete(1.0, tk.END)
                script_text.insert(1.0, "No valid script found for this methodology.\n\nPlease go back and click 'Start Generate Model Training Script' first.")
                script_text.config(state=tk.DISABLED)
        else:
            error_label = ttk.Label(
                info_label_frame, 
                text="No methodology selected. Please go back and predict a methodology first.",
                foreground="red"
            )
            error_label.pack(side=tk.LEFT)
            script_text.config(state=tk.NORMAL)
            script_text.delete(1.0, tk.END)
            script_text.insert(1.0, "No script selected.\n\nPlease go back and click 'Start Generate Model Training Script' first.")
            script_text.config(state=tk.DISABLED)
    
    # Docker execution function
    def execute_training_script():
        """Execute training script in Docker"""
        if not shared_state or not shared_state.methodology:
            messagebox.showwarning("Warning", "No methodology selected. Please go back and predict a methodology first.")
            return
        
        # Map methodology to script name (in case it wasn't set during page creation)
        methodology = shared_state.methodology
        script_name = map_methodology_to_script(methodology)
        
        if not script_name:
            messagebox.showerror("Error", f"Could not map methodology '{methodology}' to a script.\n\nSupported methodologies:\n- Python_QPCA + Regression\n- QPCA (python) + CNN (python)\n- Python_QPCA + XGBoost\n- Python_QFM + Regression\n- Python_QFM + CNN\n- Python_QFM + XGBoost\n- PQK(python, TensorFlow Quantum libraries) + CNN (python, TensorFlow libraries)\n- Matlab_SVDPCA + KNN")
            return
        
        # Check if this is a MATLAB script
        is_matlab_script = script_name.endswith('.m')
        
        # Update shared state with script name
        shared_state.script_name = script_name
        
        # Check if CSV file exists in upload folder
        # Try to find the most recent CSV file in upload folder
        upload_folder = Path("upload")
        csv_files = list(upload_folder.glob("*.csv"))
        if not csv_files:
            messagebox.showwarning("Warning", "No CSV file found in upload folder.\n\nPlease process images first to create the CSV file.")
            return
        
        # Use the most recent CSV file
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        
        # Save current version before execution
        if version_manager and shared_state and shared_state.script_name:
            script_path = Path("temporary_model_training_script") / script_name
            if script_path.exists():
                try:
                    with open(script_path, 'r', encoding='utf-8') as f:
                        script_content = f.read()
                    # Save version before execution
                    current_version = getattr(popup, 'script_version', 1)
                    version_manager.save_version(shared_state.script_name, script_content, "pre_execution", current_version)
                except Exception as e:
                    print(f"Warning: Could not save version before execution: {e}")
        
        # Confirm execution
        if is_matlab_script:
            response = messagebox.askyesno(
                "Confirm Execution",
                f"Execute MATLAB training script?\n\n"
                f"Script: {script_name}\n"
                f"Dataset: {csv_file}\n"
            )
        else:
            response = messagebox.askyesno(
                "Confirm Execution",
                f"Execute training script in Docker?\n\n"
                f"Script: {script_name}\n"
                f"Dataset: {csv_file}\n\n"
                f"Maximum execution time: 24 hours"
            )
        
        if not response:
            return
        
        # Disable buttons during execution
        train_btn.config(state=tk.DISABLED)
        if status_text_callback:
            status_text_callback("")
        
        if is_matlab_script:
            if status_text_callback:
                status_text_callback("Starting MATLAB execution...\n\n")
        else:
            if status_text_callback:
                status_text_callback("Starting Docker execution...\n\n")
        popup.update_idletasks()
        
        # MATLAB execution function
        def matlab_thread():
            try:
                import subprocess
                from pathlib import Path
                from datetime import datetime
                
                project_root = Path(".")
                debug_log_path = project_root / "debug.log"
                
                def write_to_log(message, also_display=True):
                    """Write message to debug.log file"""
                    try:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entry = f"[{timestamp}] {message}\n"
                        with open(debug_log_path, 'a', encoding='utf-8', errors='replace') as log_file:
                            log_file.write(log_entry)
                    except Exception as e:
                        pass
                
                def update_status_from_matlab(line):
                    """Update status from MATLAB output and log to file"""
                    if line:
                        write_to_log(line, also_display=False)
                        # Filter out sensitive information that should not be displayed
                        line_lower = line.lower().strip()
                        if (line_lower.startswith("matlab path:") or 
                            line_lower.startswith("working directory:") or 
                            line_lower.startswith("license:")):
                            # Skip displaying these lines
                            return
                        # Sanitize line before displaying to hide personal information
                        sanitized_line = sanitize_line_for_display(line)
                        if status_text_callback:
                            popup.after(0, lambda l=sanitized_line: status_text_callback(l + "\n"))
                
                # Get script path from temporary folder
                temp_script_folder = Path(project_root / "temporary_model_training_script")
                if not temp_script_folder.exists():
                    error_msg = f"Temporary script folder not found: {temp_script_folder}"
                    sanitized_error = sanitize_line_for_display(error_msg)
                    if status_text_callback:
                        status_text_callback(f"Error: {sanitized_error}\n")
                        status_text_callback("Please go back and click 'Start Generate Model Training Script' first.\n")
                    popup.after(0, lambda: train_btn.config(state=tk.NORMAL))
                    return
                
                matlab_script_path = temp_script_folder / script_name
                if not matlab_script_path.exists():
                    error_msg = f"MATLAB script not found: {matlab_script_path}"
                    sanitized_error = sanitize_line_for_display(error_msg)
                    if status_text_callback:
                        status_text_callback(f"Error: {sanitized_error}\n")
                    popup.after(0, lambda: train_btn.config(state=tk.NORMAL))
                    return
                
                # Update MATLAB script with correct dataset path
                # Read the script, update DATASET_PATH, and write back
                try:
                    with open(matlab_script_path, 'r', encoding='utf-8') as f:
                        script_content = f.read()
                    
                    # Update DATASET_PATH to point to the CSV file
                    import re
                    csv_path_abs = csv_file.absolute()
                    # Convert Windows path to MATLAB-friendly format
                    csv_path_matlab = str(csv_path_abs).replace('\\', '/')
                    
                    # Replace DATASET_PATH in script
                    pattern = r"CFG\.DATASET_PATH\s*=\s*['\"]([^'\"]+)['\"]"
                    replacement = f"CFG.DATASET_PATH = '{csv_path_matlab}'"
                    script_content = re.sub(pattern, replacement, script_content)
                    
                    # Update output path
                    output_path_abs = Path(project_root / "output").absolute()
                    output_path_matlab = str(output_path_abs).replace('\\', '/')
                    pattern = r"CFG\.path_saving_plot\s*=\s*['\"]([^'\"]+)['\"]"
                    replacement = f"CFG.path_saving_plot = '{output_path_matlab}'"
                    script_content = re.sub(pattern, replacement, script_content)
                    
                    # Write updated script back
                    with open(matlab_script_path, 'w', encoding='utf-8') as f:
                        f.write(script_content)
                except Exception as e:
                    error_msg = f"Failed to update MATLAB script paths: {str(e)}"
                    sanitized_error = sanitize_line_for_display(error_msg)
                    if status_text_callback:
                        status_text_callback(f"Warning: {sanitized_error}\n")
                    write_to_log(f"[WARNING] {error_msg}", also_display=False)
                
                # Initialize log file
                write_to_log("=" * 80, also_display=False)
                write_to_log(f"MATLAB Execution Started", also_display=False)
                write_to_log(f"Script: {script_name}", also_display=False)
                write_to_log(f"Dataset: {csv_file}", also_display=False)
                write_to_log("=" * 80, also_display=False)
                
                # Execute run_matlab.py with the script path
                run_matlab_script = project_root / "matlab_env" / "run_matlab.py"
                if not run_matlab_script.exists():
                    error_msg = f"run_matlab.py not found: {run_matlab_script}"
                    sanitized_error = sanitize_line_for_display(error_msg)
                    if status_text_callback:
                        status_text_callback(f"Error: {sanitized_error}\n")
                    popup.after(0, lambda: train_btn.config(state=tk.NORMAL))
                    return
                
                # Update run_matlab.py to use the script from temporary folder
                # We'll pass the script path as an environment variable or modify run_matlab.py
                # For now, let's modify run_matlab.py to accept script path as argument
                import sys
                import os
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['PYTHONUNBUFFERED'] = '1'
                
                # Call run_matlab.py with script path
                cmd = [sys.executable, str(run_matlab_script), str(matlab_script_path)]
                
                if status_text_callback:
                    status_text_callback(f"Executing MATLAB script: {script_name}\n")
                    status_text_callback(f"Dataset: {csv_file.name}\n\n")
                popup.update_idletasks()
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env,
                    cwd=str(project_root)
                )
                
                # Stream output
                for line in process.stdout:
                    if line:
                        update_status_from_matlab(line.strip())
                
                process.wait()
                
                # Get current version before execution
                current_version = getattr(popup, 'script_version', 1)
                
                if process.returncode == 0:
                    write_to_log("=" * 80, also_display=False)
                    write_to_log(f"MATLAB Execution Completed Successfully (Exit Code: {process.returncode})", also_display=False)
                    write_to_log(f"Results saved to: output/", also_display=False)
                    write_to_log("=" * 80, also_display=False)
                    
                    # Update version history with success
                    if version_manager and shared_state and shared_state.script_name:
                        version_manager.update_execution_result(
                            shared_state.script_name, current_version, "success"
                        )
                    
                    if status_text_callback:
                        status_text_callback("\n✓ Training completed successfully!\n")
                        status_text_callback(f"Results saved to: output/\n")
                        sanitized_log_path = sanitize_path_for_display(str(debug_log_path))
                        status_text_callback(f"Log saved to: {sanitized_log_path}\n")
                    popup.after(0, lambda: messagebox.showinfo(
                        "Success", 
                        f"MATLAB training script executed successfully!\n\n"
                        f"Script: {script_name}\n"
                        f"Results saved to output folder.\n"
                        f"Log saved to: {sanitize_path_for_display(str(debug_log_path))}\n\n"
                        f"Click 'View Results' to see the performance graph."
                    ))
                else:
                    write_to_log("=" * 80, also_display=False)
                    write_to_log(f"MATLAB Execution Failed (Exit Code: {process.returncode})", also_display=False)
                    write_to_log("=" * 80, also_display=False)
                    
                    # Extract error log
                    error_log = f"MATLAB execution failed with exit code {process.returncode}"
                    
                    # Update version history with failure
                    if version_manager and shared_state and shared_state.script_name:
                        version_manager.update_execution_result(
                            shared_state.script_name, current_version, "failed", error_log
                        )
                    
                    sanitized_log_path = sanitize_path_for_display(str(debug_log_path))
                    popup.after(0, lambda: messagebox.showerror(
                        "Error", 
                        f"MATLAB training script failed with exit code {process.returncode}.\n\n"
                        f"Check status and debug.log for details.\n"
                        f"Log file: {sanitized_log_path}"
                    ))
                    
            except FileNotFoundError:
                error_msg = "Python or MATLAB not found. Please ensure Python and MATLAB are installed."
                if status_text_callback:
                    status_text_callback(f"\n✗ Error: {error_msg}\n")
                popup.after(0, lambda: messagebox.showerror("Error", error_msg))
            except Exception as e:
                error_str = str(e)
                error_msg = f"Error executing MATLAB: {error_str}"
                sanitized_error = sanitize_line_for_display(error_msg)
                if status_text_callback:
                    status_text_callback(f"\n✗ {sanitized_error}\n")
                    status_text_callback(f"Check debug.log for full error details.\n")
                popup.after(0, lambda: messagebox.showerror("Error", 
                    f"{error_msg}\n\n"
                    f"Check debug.log for full error details."))
            finally:
                popup.after(0, lambda: train_btn.config(state=tk.NORMAL))
        
        def docker_thread():
            try:
                import subprocess
                from pathlib import Path
                from datetime import datetime
                
                # Get paths
                docker_env_dir = Path("docker_env")
                project_root = Path(".")
                
                # Ensure output folder exists
                output_folder = project_root / "output"
                output_folder.mkdir(exist_ok=True)
                
                # Create debug log file
                debug_log_path = project_root / "debug.log"
                
                def write_to_log(message, also_display=True):
                    """Write message to debug.log file"""
                    try:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entry = f"[{timestamp}] {message}\n"
                        with open(debug_log_path, 'a', encoding='utf-8', errors='replace') as log_file:
                            log_file.write(log_entry)
                    except Exception as e:
                        pass
                
                def update_status_from_docker(line):
                    """Update status from Docker output and log to file"""
                    if line:
                        write_to_log(line, also_display=False)
                        # Sanitize line before displaying to hide personal information
                        sanitized_line = sanitize_line_for_display(line)
                        if status_text_callback:
                            popup.after(0, lambda l=sanitized_line: status_text_callback(l + "\n"))
                
                # Check if Docker image exists, build if not
                check_image = subprocess.run(
                    ["docker", "images", "universal-training:latest", "--format", "{{.Repository}}:{{.Tag}}"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                if not check_image.stdout.strip():
                    write_to_log("Docker image not found. Building image...", also_display=False)
                    if status_text_callback:
                        status_text_callback("Docker image not found. Building image...\n")
                    popup.update_idletasks()
                    
                    dockerfile_path = docker_env_dir / "Dockerfile"
                    build_result = subprocess.run(
                        ["docker", "build", "-t", "universal-training:latest", "-f", str(dockerfile_path), str(project_root)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=False,
                        encoding='utf-8',
                        errors='replace'
                    )
                    
                    if build_result.returncode != 0:
                        try:
                            build_output = build_result.stdout.decode('utf-8', errors='replace') if isinstance(build_result.stdout, bytes) else build_result.stdout
                        except:
                            build_output = str(build_result.stdout)
                        error_msg = f"Failed to build Docker image:\n{build_output}"
                        write_to_log(f"[ERROR] {error_msg}", also_display=False)
                        sanitized_error = sanitize_line_for_display(error_msg)
                        if status_text_callback:
                            status_text_callback(f"\n✗ {sanitized_error}\n")
                        popup.after(0, lambda: messagebox.showerror("Error", sanitized_error))
                        return
                    
                    write_to_log("Docker image built successfully!", also_display=False)
                    if status_text_callback:
                        status_text_callback("✓ Docker image built successfully!\n\n")
                    popup.update_idletasks()
                
                # Prepare Docker command
                csv_filename = csv_file.name
                dataset_path_in_docker = f"/app/data/{csv_filename}"
                
                # Mount wrapper script
                wrapper_script = docker_env_dir / "run_training_wrapper.py"
                
                # Use temporary folder for script (read-write so user can modify it)
                temp_script_folder = Path(project_root / "temporary_model_training_script")
                if not temp_script_folder.exists():
                    error_msg = f"Error: Temporary script folder not found: {temp_script_folder}"
                    sanitized_error = sanitize_line_for_display(error_msg)
                    if status_text_callback:
                        status_text_callback(f"{sanitized_error}\n")
                        status_text_callback("Please go back and click 'Start Generate Model Training Script' first.\n")
                    popup.after(0, lambda: train_btn.config(state=tk.NORMAL))
                    return
                
                docker_cmd = [
                    "docker", "run", "--rm",
                    "--name", "universal-training-run",
                    "--stop-timeout", "86400",  # 24 hours
                    "-v", f"{Path(project_root / 'upload').absolute()}:/app/data:ro",
                    "-v", f"{Path(project_root / 'output').absolute()}:/app/output:rw",
                    "-v", f"{temp_script_folder.absolute()}:/app/universal_training_script:rw",  # Read-write for user modifications
                    "-v", f"{wrapper_script.absolute()}:/app/run_training_wrapper.py:ro",
                    "-w", "/app",
                    "-e", "PYTHONUNBUFFERED=1",
                    "universal-training:latest",
                    "python", "run_training_wrapper.py", script_name, dataset_path_in_docker
                ]
                
                # Initialize log file with header (use actual paths in log file)
                write_to_log("=" * 80, also_display=False)
                write_to_log(f"Docker Execution Started", also_display=False)
                write_to_log(f"Script: {script_name}", also_display=False)
                write_to_log(f"Dataset: {dataset_path_in_docker}", also_display=False)
                write_to_log(f"Command: {' '.join(docker_cmd)}", also_display=False)
                write_to_log("=" * 80, also_display=False)
                
                # Display simplified status (command details hidden)
                if status_text_callback:
                    status_text_callback("Executing Docker command...\n\n")
                popup.update_idletasks()
                
                # Set environment to use UTF-8 encoding
                import os as os_module
                env = os_module.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['PYTHONUNBUFFERED'] = '1'
                env['LC_ALL'] = 'C.UTF-8'
                env['LANG'] = 'C.UTF-8'
                
                # Execute Docker command - read as bytes to avoid encoding issues
                try:
                    process = subprocess.Popen(
                        docker_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=False,
                        bufsize=1,
                        env=env
                    )
                except Exception as e:
                    error_str = str(e).encode('utf-8', errors='replace').decode('utf-8') if isinstance(str(e), bytes) else str(e)
                    raise Exception(f"Failed to start Docker process: {error_str}")
                
                # Stream output with proper UTF-8 encoding handling
                import io
                decoder = io.TextIOWrapper(process.stdout, encoding='utf-8', errors='replace', line_buffering=True)
                
                try:
                    for line in decoder:
                        if line:
                            decoded_line = line.strip()
                            if decoded_line:
                                update_status_from_docker(decoded_line)
                except (UnicodeDecodeError, UnicodeError) as e:
                    decoder.close()
                    try:
                        remaining_output, _ = process.communicate()
                        if remaining_output:
                            decoded = remaining_output.decode('utf-8', errors='replace')
                            for line in decoded.split('\n'):
                                if line.strip():
                                    update_status_from_docker(line.strip())
                    except Exception:
                        update_status_from_docker(f"[Encoding error occurred, but execution may have succeeded]")
                except Exception as e:
                    try:
                        decoder.close()
                        remaining_output, _ = process.communicate()
                        if remaining_output:
                            decoded = remaining_output.decode('utf-8', errors='replace')
                            for line in decoded.split('\n'):
                                if line.strip():
                                    update_status_from_docker(line.strip())
                    except Exception:
                        pass
                
                process.wait()
                
                # Get current version before execution
                current_version = getattr(popup, 'script_version', 1)
                
                if process.returncode == 0:
                    write_to_log("=" * 80, also_display=False)
                    write_to_log(f"Docker Execution Completed Successfully (Exit Code: {process.returncode})", also_display=False)
                    write_to_log(f"Results saved to: output/", also_display=False)
                    write_to_log("=" * 80, also_display=False)
                    
                    # Update version history with success
                    if version_manager and shared_state and shared_state.script_name:
                        version_manager.update_execution_result(
                            shared_state.script_name, current_version, "success"
                        )
                    
                    if status_text_callback:
                        status_text_callback("\n✓ Training completed successfully!\n")
                        status_text_callback(f"Results saved to: output/\n")
                        sanitized_log_path = sanitize_path_for_display(str(debug_log_path))
                        status_text_callback(f"Log saved to: {sanitized_log_path}\n")
                    popup.after(0, lambda: messagebox.showinfo(
                        "Success", 
                        f"Training script executed successfully!\n\n"
                        f"Script: {script_name}\n"
                        f"Results saved to output folder.\n"
                        f"Log saved to: {sanitize_path_for_display(str(debug_log_path))}\n\n"
                        f"Click 'View Results' to see the performance graph."
                    ))
                else:
                    write_to_log("=" * 80, also_display=False)
                    write_to_log(f"Docker Execution Failed (Exit Code: {process.returncode})", also_display=False)
                    write_to_log("=" * 80, also_display=False)
                    
                    # Extract error log
                    error_log = f"Docker execution failed with exit code {process.returncode}"
                    
                    # Update version history with failure
                    if version_manager and shared_state and shared_state.script_name:
                        version_manager.update_execution_result(
                            shared_state.script_name, current_version, "failed", error_log
                        )
                    
                    sanitized_log_path = sanitize_path_for_display(str(debug_log_path))
                    popup.after(0, lambda: messagebox.showerror(
                        "Error", 
                        f"Training script failed with exit code {process.returncode}.\n\n"
                        f"Check status and debug.log for details.\n"
                        f"Log file: {sanitized_log_path}"
                    ))
                    
            except FileNotFoundError:
                error_msg = "Docker not found. Please install Docker and ensure it's running."
                if status_text_callback:
                    status_text_callback(f"\n✗ Error: {error_msg}\n")
                popup.after(0, lambda: messagebox.showerror("Error", error_msg))
            except UnicodeDecodeError as e:
                error_msg = f"Encoding error: {type(e).__name__}. This may be due to special characters in the output. The execution may have succeeded - please check the output folder for results."
                if status_text_callback:
                    status_text_callback(f"\n⚠ Warning: {error_msg}\n")
                popup.after(0, lambda: messagebox.showwarning("Encoding Warning", 
                    f"An encoding error occurred while reading Docker output.\n\n"
                    f"This is usually harmless and the script may have executed successfully.\n\n"
                    f"Please check the 'output' folder for the performance graph and results.\n"
                    f"Check debug.log for full details."))
            except Exception as e:
                try:
                    error_str = str(e)
                except UnicodeDecodeError:
                    try:
                        error_str = repr(e).encode('utf-8', errors='replace').decode('utf-8')
                    except:
                        error_str = "An error occurred (encoding issue prevented error message display)"
                
                error_msg = f"Error executing Docker: {error_str}"
                sanitized_error = sanitize_line_for_display(error_msg)
                if status_text_callback:
                    status_text_callback(f"\n✗ {sanitized_error}\n")
                    status_text_callback(f"Check debug.log for full error details.\n")
                popup.after(0, lambda: messagebox.showerror("Error", 
                    f"{sanitized_error}\n\n"
                    f"Check debug.log for full error details."))
            finally:
                popup.after(0, lambda: train_btn.config(state=tk.NORMAL))
        
        # Run execution in separate thread (MATLAB or Docker)
        if is_matlab_script:
            thread = threading.Thread(target=matlab_thread, daemon=True)
        else:
            thread = threading.Thread(target=docker_thread, daemon=True)
        thread.start()
    
    # Start Training button
    train_btn = ttk.Button(
        main_frame, 
        text="🤖 Start Training (Docker)", 
        command=execute_training_script
    )
    train_btn.grid(row=3, column=0, pady=(10, 0))
    
    # Initial script content update
    update_script_content()
    
    # Store update function, script_text widget, and version increment function for external calls
    popup.update_script_content = update_script_content
    popup.script_text = script_text  # Store reference to script_text widget for LLM updates
    popup.increment_version = increment_version  # Store version increment function for LLM updates
    popup.window = popup  # Store popup window reference for scheduling callbacks in popup's event loop
    
    return popup


def create_show_script_and_execute_page(parent_frame, on_back=None, on_next=None, shared_state=None, update_callback=None):
    """Create show script and execute page content in the given parent frame
    
    Args:
        parent_frame: Parent frame to create the page in
        on_back: Optional callback function to call when "Back" button is clicked
        on_next: Optional callback function to call when "View Results" button is clicked
        shared_state: Optional shared state object to get methodology and script info
        update_callback: Optional callback to register page update function
    """
    if not GUI_AVAILABLE:
        error_label = ttk.Label(parent_frame, text="GUI not available.")
        error_label.pack()
        return
    
    # Get root window for callbacks
    root = parent_frame.winfo_toplevel()
    
    # Store popup window reference and LLM widget reference
    script_popup = None
    llm_widget_ref = None  # Store reference to LLM widget for updating script_text_widget
    main_frame = ttk.Frame(parent_frame, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, text="Terminal", font=("Arial", 16, "bold"))
    title_label.grid(row=0, column=0, pady=(0, 20))
    
    # Right column: Execution status (only one column now)
    right_frame = ttk.Frame(main_frame, padding="10")
    right_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    main_frame.rowconfigure(1, weight=1)
    main_frame.columnconfigure(0, weight=1)
    
    # Function to sanitize paths for display (hide personal information)
    def sanitize_path_for_display(path_str):
        """Replace absolute paths with relative paths or placeholders for display"""
        import os
        try:
            # Get the project root (current working directory)
            project_root = Path(".").absolute()
            project_root_str = str(project_root)
            
            # Replace project root with [PROJECT_ROOT]
            if path_str.startswith(project_root_str):
                sanitized = path_str.replace(project_root_str, "[PROJECT_ROOT]")
                # Normalize path separators for display
                sanitized = sanitized.replace("\\", "/")
                return sanitized
            
            # If it's a Windows path with user directory, try to hide username
            if "\\Users\\" in path_str or "/Users/" in path_str:
                # Try to extract just the relative part from project root
                try:
                    path_obj = Path(path_str)
                    if path_obj.is_absolute():
                        rel_path = path_obj.relative_to(project_root)
                        return f"[PROJECT_ROOT]/{rel_path}".replace("\\", "/")
                except:
                    pass
            
            # Fallback: just normalize separators
            return path_str.replace("\\", "/")
        except:
            # If anything fails, return original
            return path_str
    
    # Function to sanitize command for display
    def sanitize_command_for_display(cmd_list):
        """Sanitize Docker command list for display"""
        sanitized = []
        for item in cmd_list:
            if item.startswith("docker"):
                sanitized.append(item)
            elif item.startswith("-") or item.startswith("--"):
                sanitized.append(item)
            elif "/" in item or "\\" in item or ":" in item:
                # This looks like a path, sanitize it
                sanitized.append(sanitize_path_for_display(item))
            else:
                sanitized.append(item)
        return sanitized
    
    # Function to sanitize a line of text that may contain paths
    def sanitize_line_for_display(line):
        """Sanitize a line of text by replacing any paths with sanitized versions"""
        import re
        try:
            # Pattern to match Windows paths (C:\... or \\...)
            # Also matches Unix-style paths that look like absolute paths
            # Matches paths that contain drive letters, backslashes, or forward slashes with colons
            # More comprehensive pattern that handles paths with spaces and various characters
            path_patterns = [
                r'[A-Za-z]:[\\/][^\s:<>"|?*]+(?:\s+[^\s:<>"|?*]+)*',  # Windows drive paths (C:\...)
                r'[\\/][^\s:<>"|?*]+(?:\s+[^\s:<>"|?*]+)*',  # Absolute paths starting with / or \
            ]
            
            sanitized_line = line
            for pattern in path_patterns:
                def replace_path(match):
                    path_str = match.group(0)
                    return sanitize_path_for_display(path_str)
                
                sanitized_line = re.sub(pattern, replace_path, sanitized_line)
            
            # Also handle paths that might be in messages like "Plot saved: C:\..."
            # Pattern for "Plot saved: <path>" or similar messages
            plot_saved_pattern = r'(Plot saved:\s*)([A-Za-z]:[\\/][^\s]+|[\\/][^\s]+)'
            def replace_plot_path(match):
                prefix = match.group(1)
                path_str = match.group(2)
                return prefix + sanitize_path_for_display(path_str)
            
            sanitized_line = re.sub(plot_saved_pattern, replace_plot_path, sanitized_line)
            
            # Handle paths in messages like "Using script path from command-line argument: <path>"
            # or "Changed working directory to: <path>"
            path_in_message_pattern = r'((?:Using|Changed|Script|Working|Current|Original|MATLAB|Script directory|MATLAB script path|MATLAB path|License):\s*)([A-Za-z]:[\\/][^\s]+|[\\/][^\s]+)'
            def replace_message_path(match):
                prefix = match.group(1)
                path_str = match.group(2)
                return prefix + sanitize_path_for_display(path_str)
            
            sanitized_line = re.sub(path_in_message_pattern, replace_message_path, sanitized_line)
            
            return sanitized_line
        except:
            # If anything fails, return original
            return line
    
    # Status update callback function
    def update_status(message):
        """Update status text area."""
        status_text.insert(tk.END, message)
        status_text.see(tk.END)
        root.update_idletasks()
    
    # Create a horizontal PanedWindow to split Execution Status and LLM Assistant side by side
    paned = ttk.PanedWindow(right_frame, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True)
    
    # Left side: Execution Status
    status_frame = ttk.LabelFrame(paned, text="Execution Status", padding="10")
    paned.add(status_frame, weight=1)
    status_frame.columnconfigure(0, weight=1)
    status_frame.rowconfigure(0, weight=1)
    
    status_text = scrolledtext.ScrolledText(status_frame, height=15, width=60, wrap=tk.WORD)
    status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Right side: LLM Chat area
    # Create a dummy script_text widget for LLM (it will be updated to use popup's script_text)
    dummy_script_text = None
    if LLM_AVAILABLE:
        # Create a placeholder text widget for LLM initialization
        # The LLM widget will be updated to use the popup's script_text when popup is created
        llm_frame, llm_chat_text, llm_input_entry, llm_send_btn = create_llm_chat_widget(
            paned, None, shared_state, root, version_callback=None
        )
                    # Store reference to LLM widget for later updates
        llm_widget_ref = {
            'frame': llm_frame,
            'chat_text': llm_chat_text,
            'input_entry': llm_input_entry,
            'send_btn': llm_send_btn
        }
        # Store popup window reference in LLM frame for version access
        llm_frame.popup_window = script_popup
        # Unpack the frame (it was packed by create_llm_chat_widget) and add to PanedWindow
        llm_frame.pack_forget()
        paned.add(llm_frame, weight=1)
    else:
        # Fallback: Create a placeholder frame if LLM is not available
        llm_frame = ttk.LabelFrame(paned, text="🤖 LLM Assistant (Not Available)", padding="10")
        paned.add(llm_frame, weight=1)
        llm_frame.columnconfigure(0, weight=1)
        llm_frame.rowconfigure(0, weight=1)
        error_label = ttk.Label(llm_frame, text="LLM integration module not found.", foreground="gray")
        error_label.grid(row=0, column=0)
    
    # Function to create and open popup window
    def open_script_editor_popup():
        """Open the script editor popup window"""
        nonlocal script_popup, llm_widget_ref
        
        def update_llm_reference():
            """Update LLM widget reference to popup's script_text"""
            if LLM_AVAILABLE and llm_widget_ref and script_popup and script_popup.winfo_exists():
                if hasattr(script_popup, 'script_text'):
                    # Update the LLM widget's script_text_widget reference
                    llm_widget_ref['frame'].script_text_widget = script_popup.script_text
                    # Also store a callback to refresh the script display
                    if hasattr(script_popup, 'update_script_content'):
                        llm_widget_ref['frame'].refresh_script_callback = script_popup.update_script_content
                    # Store version increment callback for LLM modifications
                    if hasattr(script_popup, 'increment_version'):
                        llm_widget_ref['frame'].version_callback = script_popup.increment_version
                    # Store popup window reference for scheduling callbacks in popup's event loop
                    if hasattr(script_popup, 'window'):
                        llm_widget_ref['frame'].popup_window = script_popup.window
                    print(f"DEBUG: Updated LLM widget reference to popup's script_text")
        
        if script_popup is None or not script_popup.winfo_exists():
            script_popup = create_script_editor_popup(
                root, shared_state, update_status, sanitize_path_for_display, sanitize_line_for_display
            )
            # Update reference after popup is created (use after() to ensure it's fully initialized)
            root.after(50, update_llm_reference)
            # Store popup reference in LLM frame for version access
            if llm_widget_ref and llm_widget_ref.get('frame'):
                llm_widget_ref['frame'].popup_window = script_popup
        else:
            # Bring existing popup to front
            script_popup.lift()
            script_popup.focus_force()
            # Update LLM widget reference if popup already exists
            update_llm_reference()
            # Ensure popup reference is set
            if llm_widget_ref and llm_widget_ref.get('frame'):
                llm_widget_ref['frame'].popup_window = script_popup
    
    # Function to update page content based on shared state
    def update_page_content():
        """Update page content based on current shared state"""
        # Open popup window when page is shown
        open_script_editor_popup()
        
        # Ensure LLM widget reference is updated after popup opens
        # Use after() to ensure popup is fully created
        root.after(100, lambda: open_script_editor_popup() if script_popup else None)
        
        # Update popup content if it exists
        if script_popup and hasattr(script_popup, 'update_script_content'):
            script_popup.update_script_content()
    
    # Note: update_page_content() is NOT called here during initialization
    # It will be called automatically when the page is shown via the update_callback
    # registered below. This prevents the popup from opening at startup.
    
    # Button frame (below the status area)
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=2, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
    button_frame.columnconfigure(1, weight=1)
    
    # Back button
    if on_back:
        def go_back():
            """Navigate back to previous step"""
            on_back()
        
        back_btn = ttk.Button(button_frame, text="⬅️ Back", command=go_back)
        back_btn.grid(row=0, column=0, sticky=tk.W)
    
    # View Results button
    if on_next:
        def go_next():
            """Navigate to next step (performance viewer)"""
            on_next()
        
        next_btn = ttk.Button(button_frame, text="➡️ View Results", command=go_next)
        next_btn.grid(row=0, column=1, sticky=tk.E)
    
    # Register update function if callback provided
    if update_callback:
        update_callback(update_page_content)


def create_gui():
    """Create standalone GUI application"""
    if not GUI_AVAILABLE:
        print("GUI not available.")
        return
    
    root = tk.Tk()
    root.title("Show Script and Execute")
    root.geometry("1200x800")
    
    create_show_script_and_execute_page(root)
    root.mainloop()


if __name__ == "__main__":
    create_gui()
