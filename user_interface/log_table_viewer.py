# -*- coding: utf-8 -*-
"""
Log Table Viewer GUI
Displays a table showing script versions, error logs, and modifications.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
from typing import Optional

try:
    from log_history_helper_functions.version_history import VersionHistoryManager
    from log_history_helper_functions.log_parser import LogParser
    from log_history_helper_functions.diff_calculator import DiffCalculator
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    MODULES_AVAILABLE = False

try:
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


def create_log_table_viewer_page(parent_frame, on_back=None, shared_state=None, update_callback=None):
    """Create log table viewer page content in the given parent frame
    
    Args:
        parent_frame: Parent frame to create the page in
        on_back: Optional callback function to call when "Back" button is clicked
        shared_state: Optional shared state object to get script name
        update_callback: Optional callback to register page update function
    """
    if not GUI_AVAILABLE:
        error_label = ttk.Label(parent_frame, text="GUI not available.")
        error_label.pack()
        return
    
    if not MODULES_AVAILABLE:
        error_label = ttk.Label(parent_frame, text="Required modules not available. Please check imports.")
        error_label.pack()
        return
    
    # Get root window for callbacks
    root = parent_frame.winfo_toplevel()
    
    # Initialize managers
    version_manager = VersionHistoryManager()
    log_parser = LogParser()
    diff_calculator = DiffCalculator()
    
    # Function to reload metadata (for refresh)
    def reload_metadata():
        """Reload version metadata from disk and recover any missing versions"""
        try:
            # Reload metadata from disk
            version_manager.metadata = version_manager._load_metadata()
            print(f"DEBUG: Reloaded metadata, found {len(version_manager.metadata)} scripts")
            
            # CRITICAL: Recover any missing versions from version files
            # This ensures that even if metadata is incomplete, all version files are represented
            script_name = script_var.get()
            if script_name:
                recovered = version_manager.recover_missing_versions(script_name)
                if recovered > 0:
                    print(f"DEBUG: Recovered {recovered} missing version(s) for {script_name} during reload")
            else:
                # Recover for all scripts if no specific script selected
                for script in list(version_manager.metadata.keys()):
                    recovered = version_manager.recover_missing_versions(script)
                    if recovered > 0:
                        print(f"DEBUG: Recovered {recovered} missing version(s) for {script} during reload")
        except Exception as e:
            print(f"Error reloading metadata: {e}")
            import traceback
            traceback.print_exc()
    
    # Main frame
    main_frame = ttk.Frame(parent_frame, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, text="Script Modification History", font=("Arial", 16, "bold"))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    
    # Instructions
    info_text = "View script modification history, error logs, and version differences."
    info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT)
    info_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
    
    # Script selection (if multiple scripts exist)
    script_frame = ttk.LabelFrame(main_frame, text="Script Selection", padding="10")
    script_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
    
    script_var = tk.StringVar()
    if shared_state and shared_state.script_name:
        script_var.set(shared_state.script_name)
    
    ttk.Label(script_frame, text="Script:").grid(row=0, column=0, padx=(0, 10))
    script_combo = ttk.Combobox(script_frame, textvariable=script_var, width=40, state="readonly")
    script_combo.grid(row=0, column=1, padx=(0, 10))
    
    refresh_btn = ttk.Button(script_frame, text="🔄 Refresh", command=lambda: refresh_table())
    refresh_btn.grid(row=0, column=2)
    
    # Table frame
    table_frame = ttk.LabelFrame(main_frame, text="Modification History", padding="10")
    table_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    table_frame.columnconfigure(0, weight=1)
    table_frame.rowconfigure(0, weight=1)
    main_frame.rowconfigure(3, weight=1)
    main_frame.columnconfigure(0, weight=1)
    
    # Create treeview (table)
    columns = ("Version", "Error Log", "User Prompt", "Modification")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
    
    # Configure columns
    tree.heading("Version", text="Number of Modification")
    tree.heading("Error Log", text="Error Log")
    tree.heading("User Prompt", text="User Prompt")
    tree.heading("Modification", text="Modification")
    
    tree.column("Version", width=150, anchor=tk.CENTER)
    tree.column("Error Log", width=200)
    tree.column("User Prompt", width=250)
    tree.column("Modification", width=250)
    
    # Scrollbars
    v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    
    # Grid layout
    tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    # Store full data for each row
    row_data = {}
    
    def refresh_table():
        """Refresh the table with current data"""
        # Reload metadata to get latest versions (this also recovers missing versions)
        reload_metadata()
        
        # Clear existing data
        for item in tree.get_children():
            tree.delete(item)
        row_data.clear()
        
        script_name = script_var.get()
        if not script_name:
            return
        
        # CRITICAL: Ensure all version files are in metadata before displaying
        # This is a safety check in case recovery didn't work
        version_manager._ensure_all_file_versions_in_metadata(script_name)
        
        # Get all versions
        versions = version_manager.get_all_versions(script_name)
        print(f"DEBUG: Found {len(versions)} versions for {script_name}: {[v['version'] for v in versions]}")
        
        if not versions:
            # Show message if no versions found
            tree.insert("", tk.END, values=(
                "No versions",
                "No version history available",
                "N/A",
                "N/A"
            ))
            return
        
        # Get latest execution from log
        latest_execution = log_parser.get_latest_execution(script_name)
        
        # Populate table
        for i, version_info in enumerate(versions):
            version_num = version_info["version"]
            
            # Error log
            error_log = version_info.get("error_log", "")
            if not error_log and version_info.get("execution_result") == "failed":
                # Try to get from log parser
                errors = log_parser.get_errors_for_script(script_name)
                if errors:
                    # Find error closest to this version's timestamp
                    version_time = version_info.get("execution_timestamp")
                    if version_time:
                        for err in errors:
                            if err["timestamp"] and err["timestamp"] >= version_time:
                                error_log = err["error_log"]
                                break
                    if not error_log and errors:
                        error_log = errors[0]["error_log"]
            
            # Truncate error log for display (handle None case)
            if error_log and isinstance(error_log, str):
                error_display = error_log[:100] + "..." if len(error_log) > 100 else error_log
            else:
                error_display = "No errors" if version_info.get("execution_result") == "success" else "N/A"
            
            # User prompt (for LLM modifications)
            user_prompt = version_info.get("user_prompt", "")
            if user_prompt and isinstance(user_prompt, str):
                # Truncate prompt for display
                prompt_display = user_prompt[:80] + "..." if len(user_prompt) > 80 else user_prompt
            else:
                prompt_display = "N/A"
            
            # Modification diff
            modification_display = "N/A"
            if i > 0:
                # Compare with previous version
                prev_version = versions[i - 1]["version"]
                prev_content = version_manager.get_version_content(script_name, prev_version)
                curr_content = version_manager.get_version_content(script_name, version_num)
                
                if prev_content and curr_content:
                    diff = diff_calculator.calculate_diff(
                        prev_content, curr_content, prev_version, version_num
                    )
                    modification_display = diff_calculator.get_diff_preview(prev_content, curr_content)
                    # Store full diff
                    row_data[version_num] = {
                        "error_log": error_log or "",
                        "modification": diff,
                        "prev_version": prev_version,
                        "user_prompt": user_prompt or ""
                    }
            else:
                # First version
                modification_display = "Initial version"
                row_data[version_num] = {
                    "error_log": error_log or "",
                    "modification": "Initial version",
                    "prev_version": None,
                    "user_prompt": user_prompt or ""
                }
            
            # Insert row
            item_id = tree.insert("", tk.END, values=(
                f"Version {version_num}",
                error_display,
                prompt_display,
                modification_display
            ))
            
            # Color code based on execution result
            if version_info.get("execution_result") == "failed":
                tree.set(item_id, "Error Log", "❌ " + error_display)
            elif version_info.get("execution_result") == "success":
                tree.set(item_id, "Error Log", "✅ " + error_display)
    
    def on_row_select(event):
        """Handle row selection - show details in popup"""
        selection = tree.selection()
        if not selection:
            return
        
        item = tree.item(selection[0])
        values = item['values']
        if not values:
            return
        
        # Extract version number
        version_str = values[0]
        version_num = int(version_str.split()[-1])
        
        script_name = script_var.get()
        if not script_name:
            return
        
        # Get stored data
        data = row_data.get(version_num, {})
        
        # Create popup
        popup = tk.Toplevel(root)
        popup.title(f"Version {version_num} Details")
        popup.geometry("800x600")
        
        # Notebook for tabs
        notebook = ttk.Notebook(popup)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # User Prompt tab (if available)
        user_prompt = data.get("user_prompt", "")
        if user_prompt:
            prompt_frame = ttk.Frame(notebook)
            notebook.add(prompt_frame, text="User Prompt")
            
            prompt_text = scrolledtext.ScrolledText(prompt_frame, wrap=tk.WORD, font=("Arial", 10))
            prompt_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            prompt_text.insert(1.0, user_prompt)
            prompt_text.config(state=tk.DISABLED)
        
        # Error log tab
        error_frame = ttk.Frame(notebook)
        notebook.add(error_frame, text="Error Log")
        
        error_text = scrolledtext.ScrolledText(error_frame, wrap=tk.WORD, font=("Consolas", 9))
        error_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        error_text.insert(1.0, data.get("error_log", "No error log available"))
        error_text.config(state=tk.DISABLED)
        
        # Modification tab
        mod_frame = ttk.Frame(notebook)
        notebook.add(mod_frame, text="Modification Diff")
        
        mod_text = scrolledtext.ScrolledText(mod_frame, wrap=tk.NONE, font=("Consolas", 9))
        mod_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        mod_text.insert(1.0, data.get("modification", "No modification details available"))
        mod_text.config(state=tk.DISABLED)
    
    # Bind double-click to show details
    tree.bind("<Double-1>", on_row_select)
    
    def populate_script_list():
        """Populate script combo box with available scripts"""
        # Reload metadata first
        reload_metadata()
        
        scripts = list(version_manager.metadata.keys())
        if scripts:
            script_combo['values'] = scripts
            if not script_var.get() and scripts:
                script_var.set(scripts[0])
            refresh_table()
        else:
            script_combo['values'] = []
            if shared_state and shared_state.script_name:
                script_combo['values'] = [shared_state.script_name]
                script_var.set(shared_state.script_name)
                refresh_table()
    
    # Button frame
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=4, column=0, columnspan=2, pady=(10, 0))
    
    # Back button
    if on_back:
        def go_back():
            """Navigate back to previous step"""
            on_back()
        
        back_btn = ttk.Button(button_frame, text="⬅️ Back", command=go_back)
        back_btn.pack(side=tk.LEFT, padx=(0, 10))
    
    # View Details button
    view_details_btn = ttk.Button(button_frame, text="📋 View Details", command=lambda: on_row_select(None))
    view_details_btn.pack(side=tk.LEFT)
    
    # Function to update page content when shown
    def update_page_content():
        """Update page content when page is shown"""
        # CRITICAL: Recover missing versions before loading
        script_name = script_var.get() if script_var.get() else (shared_state.script_name if shared_state and shared_state.script_name else None)
        if script_name:
            version_manager.recover_missing_versions(script_name)
            version_manager._ensure_all_file_versions_in_metadata(script_name)
        
        reload_metadata()
        populate_script_list()
        if script_var.get():
            refresh_table()
    
    # Register update function if callback provided
    if update_callback:
        update_callback(update_page_content)
    
    # Initialize - ensure recovery happens on first load
    if shared_state and shared_state.script_name:
        version_manager.recover_missing_versions(shared_state.script_name)
        version_manager._ensure_all_file_versions_in_metadata(shared_state.script_name)
    
    populate_script_list()
    if script_var.get():
        refresh_table()


def create_gui():
    """Create standalone GUI application for log table viewer"""
    if not GUI_AVAILABLE:
        print("GUI not available.")
        return
    
    root = tk.Tk()
    root.title("Log Table Viewer")
    root.geometry("1000x700")
    
    create_log_table_viewer_page(root)
    root.mainloop()


if __name__ == "__main__":
    create_gui()

