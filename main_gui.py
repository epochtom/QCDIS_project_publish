# -*- coding: utf-8 -*-
"""
Main GUI Launcher
Provides sequential page navigation between different application pages.
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Import GUI functions from other modules
try:
    from user_interface.prompt_predictor_gui import create_prompt_predictor_page
    from user_interface.image_to_csv import create_image_to_csv_page
    from user_interface.show_script_and_execute import create_show_script_and_execute_page
    from user_interface.performance_viewer import create_performance_viewer_page
    from user_interface.log_table_viewer import create_log_table_viewer_page
    from log_history_helper_functions.version_history import VersionHistoryManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all GUI modules are in the 'user_interface' folder.")
    raise


class SharedState:
    """Shared state object to pass data between pages"""
    def __init__(self):
        self.methodology = None  # Store the predicted methodology from page 1
        self.script_name = None  # Store the script name for Docker execution


def create_main_gui():
    """Create the main GUI with sequential page navigation"""
    # Clear version history on startup (fresh start for each session)
    # Auto-recovery will work DURING program execution if versions go missing
    try:
        version_manager = VersionHistoryManager(auto_recover=False)  # Disable auto-recovery on startup since we're clearing
        version_manager.clear_all_history()
        print("Version history cleared on startup.")
    except Exception as e:
        print(f"Warning: Could not clear version history on startup: {e}")
    
    root = tk.Tk()
    root.title("QCDIS")
    root.geometry("1920x1080")
    
    # Shared state for passing data between pages
    shared_state = SharedState()
    
    # Container frame for pages
    container = ttk.Frame(root)
    container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create page frames
    page1_frame = ttk.Frame(container)
    page2_frame = ttk.Frame(container)
    page3_frame = ttk.Frame(container)
    page4_frame = ttk.Frame(container)
    page5_frame = ttk.Frame(container)
    
    # Store page update functions
    page_update_functions = {}
    
    # Function to show a page
    def show_page(page_frame):
        """Hide all pages and show the specified page"""
        page1_frame.pack_forget()
        page2_frame.pack_forget()
        page3_frame.pack_forget()
        page4_frame.pack_forget()
        page5_frame.pack_forget()
        page_frame.pack(fill=tk.BOTH, expand=True)
        
        # Update page content if it has an update function
        if page_frame in page_update_functions:
            print(f"DEBUG: Calling update function for page {page_frame}")
            try:
                page_update_functions[page_frame]()
            except Exception as e:
                print(f"DEBUG: Error calling update function: {e}")
                import traceback
                traceback.print_exc()
    
    # Create pages with navigation callbacks
    create_prompt_predictor_page(
        page1_frame, 
        on_next=lambda: show_page(page2_frame),
        shared_state=shared_state
    )
    create_image_to_csv_page(
        page2_frame, 
        on_back=lambda: show_page(page1_frame),
        on_next=lambda: show_page(page3_frame),
        shared_state=shared_state
    )
    def register_page3_update(update_func):
        """Register update function for page 3"""
        page_update_functions[page3_frame] = update_func
    
    create_show_script_and_execute_page(
        page3_frame,
        on_back=lambda: show_page(page2_frame),
        on_next=lambda: show_page(page4_frame),
        shared_state=shared_state,
        update_callback=register_page3_update
    )
    create_performance_viewer_page(
        page4_frame,
        on_back=lambda: show_page(page3_frame),
        on_next=lambda: show_page(page5_frame)
    )
    def register_page5_update(update_func):
        """Register update function for page 5"""
        page_update_functions[page5_frame] = update_func
    
    create_log_table_viewer_page(
        page5_frame,
        on_back=lambda: show_page(page4_frame),
        shared_state=shared_state,
        update_callback=register_page5_update
    )
    
    # Show first page initially
    show_page(page1_frame)
    
    root.mainloop()


if __name__ == "__main__":
    create_main_gui()

