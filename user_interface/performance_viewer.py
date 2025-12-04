# -*- coding: utf-8 -*-
"""
Performance Viewer GUI
Displays performance graphs from training script execution.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import threading
import os
import glob

try:
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


def create_performance_viewer_page(parent_frame, on_back=None, on_next=None):
    """Create performance viewer page content in the given parent frame
    
    Args:
        parent_frame: Parent frame to create the page in
        on_back: Optional callback function to call when "Back" button is clicked
        on_next: Optional callback function to call when "Next" button is clicked
    """
    if not GUI_AVAILABLE:
        error_label = ttk.Label(parent_frame, text="GUI not available.")
        error_label.pack()
        return
    
    # Get root window for callbacks
    root = parent_frame.winfo_toplevel()
    
    # Main frame
    main_frame = ttk.Frame(parent_frame, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, text="Performance Results", font=("Arial", 16, "bold"))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    
    # Instructions
    info_text = "Drag and drop plot names from left to right to view them."
    info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT)
    info_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
    
    # Create two-column layout using PanedWindow
    paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
    paned.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    main_frame.rowconfigure(2, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    
    # Left column: Available plots list
    left_frame = ttk.Frame(paned, padding="10")
    paned.add(left_frame, weight=1)
    
    # Right column: Image display area
    right_frame = ttk.Frame(paned, padding="10")
    paned.add(right_frame, weight=1)
    
    # Left side: Available plots list
    plots_list_frame = ttk.LabelFrame(left_frame, text="Available Plots", padding="10")
    plots_list_frame.pack(fill=tk.BOTH, expand=True)
    plots_list_frame.columnconfigure(0, weight=1)
    plots_list_frame.rowconfigure(0, weight=1)
    
    # Listbox for available plots
    plots_listbox = tk.Listbox(plots_list_frame, font=("Arial", 10), selectmode=tk.SINGLE)
    plots_listbox.pack(fill=tk.BOTH, expand=True)
    
    # Scrollbar for listbox
    plots_scrollbar = ttk.Scrollbar(plots_list_frame, orient=tk.VERTICAL, command=plots_listbox.yview)
    plots_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    plots_listbox.config(yscrollcommand=plots_scrollbar.set)
    
    # Right side: Image display area
    image_frame = ttk.LabelFrame(right_frame, text="Plot Display (Drop here)", padding="10")
    image_frame.pack(fill=tk.BOTH, expand=True)
    image_frame.columnconfigure(0, weight=1)
    image_frame.rowconfigure(0, weight=1)
    
    # Canvas for image display (drop target)
    canvas = tk.Canvas(image_frame, bg="white", relief=tk.SUNKEN, borderwidth=2, width=400, height=300)
    canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Drop zone indicator text (will be removed when image is loaded)
    canvas.drop_text = canvas.create_text(
        200, 150,
        text="Drop plot here to view\n\nOr click on a plot name",
        font=("Arial", 12),
        fill="gray",
        justify=tk.CENTER
    )
    
    # Status label
    status_label = ttk.Label(main_frame, text="No plot selected. Drag a plot from the left to view it.", 
                            foreground="gray")
    status_label.grid(row=3, column=0, columnspan=2, pady=(10, 0))
    
    def scan_available_plots():
        """Scan output folder for available plot images"""
        plots = []
        output_path = Path("output")
        output_path.mkdir(exist_ok=True)
        
        # Look for common image formats
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']
        for ext in image_extensions:
            plots.extend(output_path.glob(ext))
            plots.extend(output_path.glob(ext.upper()))
        
        # Also check current directory
        for ext in image_extensions:
            plots.extend(Path(".").glob(ext))
            plots.extend(Path(".").glob(ext.upper()))
        
        # Remove duplicates and sort by modification time (newest first)
        plots = sorted(set(plots), key=lambda p: p.stat().st_mtime, reverse=True)
        return plots
    
    def update_plots_list():
        """Update the list of available plots"""
        plots_listbox.delete(0, tk.END)
        plots = scan_available_plots()
        
        if plots:
            for plot_path in plots:
                plots_listbox.insert(tk.END, plot_path.name)
            status_label.config(text=f"Found {len(plots)} plot(s). Drag one to view.", foreground="blue")
        else:
            plots_listbox.insert(tk.END, "No plots found")
            status_label.config(text="No plots found. Run training script first.", foreground="gray")
    
    def load_plot_image(plot_path):
        """Load and display a plot image, resized to fit the available space"""
        if not plot_path.exists():
            status_label.config(text=f"Plot not found: {plot_path}", foreground="red")
            return
        
        try:
            # Load image
            img = Image.open(plot_path)
            original_size = img.size
            
            # Get available canvas size
            root.update_idletasks()
            image_frame.update_idletasks()
            available_width = image_frame.winfo_width() - 40
            available_height = image_frame.winfo_height() - 40
            
            # If frame size not yet available, use window size as fallback
            if available_width <= 0 or available_height <= 0:
                window_width = root.winfo_width()
                window_height = root.winfo_height()
                available_width = window_width // 2 - 60  # Half for right column
                available_height = window_height - 250
            
            # Ensure minimum size
            available_width = max(available_width, 400)
            available_height = max(available_height, 300)
            
            # Calculate resize dimensions maintaining aspect ratio
            img_width, img_height = original_size
            aspect_ratio = img_width / img_height
            
            if aspect_ratio > (available_width / available_height):
                # Image is wider - fit to width
                display_width = available_width
                display_height = int(available_width / aspect_ratio)
            else:
                # Image is taller - fit to height
                display_height = available_height
                display_width = int(available_height * aspect_ratio)
            
            # Resize image
            img_resized = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_resized)
            
            # Clear canvas (including drop text)
            canvas.delete("all")
            
            # Set canvas size to match image size
            canvas.config(width=display_width, height=display_height)
            
            # Display image centered
            canvas.create_image(display_width // 2, display_height // 2, anchor=tk.CENTER, image=photo)
            canvas.image = photo  # Keep a reference
            canvas.current_plot = plot_path  # Store current plot path
            
            status_label.config(
                text=f"Displaying: {plot_path.name} ({original_size[0]}x{original_size[1]} → {display_width}x{display_height})", 
                foreground="green"
            )
        except Exception as e:
            status_label.config(text=f"Error loading plot: {str(e)}", foreground="red")
            canvas.delete("all")
            canvas.create_text(
                200, 200,
                text="Error loading plot",
                font=("Arial", 12),
                fill="red"
            )
    
    # Drag and drop variables
    drag_data = {"item": None, "dragging": False, "start_x": 0, "start_y": 0}
    
    def on_listbox_press(event):
        """Handle mouse press on listbox - start drag"""
        # Get the item under the cursor
        index = plots_listbox.nearest(event.y)
        if index >= 0:
            plots_listbox.selection_clear(0, tk.END)
            plots_listbox.selection_set(index)
            plots_listbox.activate(index)
            drag_data["item"] = plots_listbox.get(index)
            drag_data["start_x"] = event.x_root
            drag_data["start_y"] = event.y_root
            drag_data["dragging"] = False  # Will be set to True on motion
    
    def on_listbox_motion(event):
        """Handle mouse motion on listbox - check if dragging"""
        if drag_data["item"]:
            # Check if mouse has moved enough to consider it a drag
            dx = abs(event.x_root - drag_data["start_x"])
            dy = abs(event.y_root - drag_data["start_y"])
            if dx > 5 or dy > 5:  # Threshold for drag
                drag_data["dragging"] = True
                plots_listbox.config(cursor="hand2")
    
    def on_listbox_release(event):
        """Handle mouse release on listbox"""
        if drag_data["dragging"]:
            # Check if mouse is over canvas
            x, y = event.x_root, event.y_root
            canvas_x = canvas.winfo_rootx()
            canvas_y = canvas.winfo_rooty()
            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()
            
            if (canvas_x <= x <= canvas_x + canvas_w and 
                canvas_y <= y <= canvas_y + canvas_h):
                # Mouse is over canvas - drop it
                on_canvas_drop(event)
            else:
                # Not over canvas - just select
                drag_data["dragging"] = False
                plots_listbox.config(cursor="")
        else:
            # Simple click - load the plot
            selection = plots_listbox.curselection()
            if selection:
                plot_name = plots_listbox.get(selection[0])
                output_path = Path("output")
                plot_path = output_path / plot_name
                if not plot_path.exists():
                    plot_path = Path(plot_name)
                if plot_path.exists():
                    load_plot_image(plot_path)
        
        drag_data["item"] = None
        drag_data["dragging"] = False
        plots_listbox.config(cursor="")
    
    def on_listbox_double_click(event):
        """Handle double-click on listbox - load the plot directly"""
        selection = plots_listbox.curselection()
        if selection:
            plot_name = plots_listbox.get(selection[0])
            output_path = Path("output")
            plot_path = output_path / plot_name
            if not plot_path.exists():
                plot_path = Path(plot_name)
            if plot_path.exists():
                load_plot_image(plot_path)
    
    def on_canvas_drop(event):
        """Handle drop on canvas"""
        canvas.config(cursor="", bg="white")
        if drag_data["item"]:
            plot_name = drag_data["item"]
            # Find the full path
            output_path = Path("output")
            plot_path = output_path / plot_name
            if not plot_path.exists():
                plot_path = Path(plot_name)
            if plot_path.exists():
                load_plot_image(plot_path)
            drag_data["item"] = None
            drag_data["dragging"] = False
    
    def on_canvas_enter(event):
        """Handle mouse enter canvas"""
        if drag_data["dragging"]:
            canvas.config(cursor="hand2", bg="lightblue")
    
    def on_canvas_leave(event):
        """Handle mouse leave canvas"""
        canvas.config(cursor="", bg="white")
    
    def on_canvas_motion(event):
        """Track mouse motion over canvas"""
        if drag_data["dragging"]:
            canvas.config(cursor="hand2", bg="lightblue")
    
    # Bind events for drag and drop
    plots_listbox.bind('<Button-1>', on_listbox_press)
    plots_listbox.bind('<B1-Motion>', on_listbox_motion)
    plots_listbox.bind('<ButtonRelease-1>', on_listbox_release)
    plots_listbox.bind('<Double-Button-1>', on_listbox_double_click)
    
    # Canvas drop events
    canvas.bind('<Enter>', on_canvas_enter)
    canvas.bind('<Leave>', on_canvas_leave)
    canvas.bind('<Motion>', on_canvas_motion)
    canvas.bind('<ButtonRelease-1>', on_canvas_drop)
    
    def refresh_plots():
        """Refresh the list of available plots"""
        status_label.config(text="Refreshing...", foreground="blue")
        root.update_idletasks()
        update_plots_list()
        # Reload current plot if one is displayed
        if hasattr(canvas, 'current_plot') and canvas.current_plot:
            load_plot_image(canvas.current_plot)
    
    # Refresh button
    refresh_btn = ttk.Button(main_frame, text="🔄 Refresh Plots List", command=refresh_plots)
    refresh_btn.grid(row=4, column=0, columnspan=2, pady=(10, 0))
    
    # Button frame for navigation
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=5, column=0, columnspan=2, pady=(20, 0))
    
    # Back button
    if on_back:
        def go_back():
            """Navigate back to previous step"""
            on_back()
        
        back_btn = ttk.Button(button_frame, text="⬅️ Back", command=go_back)
        back_btn.pack(side=tk.LEFT, padx=(0, 10))
    
    # Next button (to log table viewer)
    if on_next:
        def go_next():
            """Navigate to next step (log table viewer)"""
            on_next()
        
        next_btn = ttk.Button(button_frame, text="➡️ View Modification History", command=go_next)
        next_btn.pack(side=tk.LEFT)
    
    # Bind window resize event to reload graph with new size
    def on_window_resize(event=None):
        """Handle window resize to reload graph with new dimensions"""
        if hasattr(canvas, 'image') and canvas.image:  # Only resize if image is already loaded
            if hasattr(canvas, 'current_plot') and canvas.current_plot:
                load_plot_image(canvas.current_plot)
    
    # Bind resize event to parent frame
    parent_frame.bind('<Configure>', on_window_resize)
    
    # Initialize: Load plots list and try to load default plot
    def initialize():
        """Initialize the viewer"""
        update_plots_list()
        # Try to load performance_summary.png by default if it exists
        default_plot = Path("output") / "performance_summary.png"
        if not default_plot.exists():
            default_plot = Path("performance_summary.png")
        if default_plot.exists():
            load_plot_image(default_plot)
    
    # Load plots initially (with a small delay to ensure window is fully rendered)
    root.after(100, initialize)


def create_gui():
    """Create standalone GUI application for performance viewer"""
    if not GUI_AVAILABLE:
        print("GUI not available.")
        return
    
    root = tk.Tk()
    root.title("Performance Viewer")
    root.geometry("800x600")
    
    create_performance_viewer_page(root)
    root.mainloop()


if __name__ == "__main__":
    create_gui()

