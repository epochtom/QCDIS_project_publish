# -*- coding: utf-8 -*-
"""
Image Preprocessing Program
Converts .jpg images from labeled folders to CSV format.

Folder structure:
- Each folder name is an integer representing the label
- Each folder contains .jpg files of that label

Output CSV format:
- First column: label (integer)
- Remaining columns: pixel0, pixel1, ..., pixel16383 (128x128 = 16,384 pixels)
"""

import os
import csv
import argparse
import threading
from PIL import Image
import numpy as np
from pathlib import Path
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk, scrolledtext
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


def process_image(image_path, target_size=(128, 128)):
    """
    Process a single image: resize to target_size and convert to grayscale.
    
    Args:
        image_path: Path to the image file
        target_size: Tuple of (width, height) for resizing (default: 128x128)
    
    Returns:
        numpy array of flattened pixel values (0-255)
    """
    try:
        # Open and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        # Resize to target size
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and flatten
        img_array = np.array(img, dtype=np.uint8)
        flattened = img_array.flatten()
        
        return flattened
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def ensure_upload_folder(output_path):
    """
    Ensure the output path is in the 'upload' folder.
    Creates the folder if it doesn't exist.
    
    Args:
        output_path: Original output path (can be just filename or full path)
    
    Returns:
        Path object pointing to the file in the upload folder
    """
    output_path = Path(output_path)
    upload_folder = Path("upload")
    
    # Create upload folder if it doesn't exist
    upload_folder.mkdir(exist_ok=True)
    
    # If path already includes upload folder, use it as is
    if output_path.parts[0] == "upload":
        return output_path
    
    # Otherwise, extract just the filename and put it in upload folder
    return upload_folder / output_path.name


def process_folders(input_folders, output_csv_path, target_size=(128, 128), progress_callback=None):
    """
    Process multiple labeled folders and create a CSV file.
    
    Args:
        input_folders: List of folder paths (folder names are integer labels)
        output_csv_path: Path to output CSV file (will be saved in 'upload' folder)
        target_size: Tuple of (width, height) for resizing (default: 128x128)
        progress_callback: Optional callback function(status_message) for progress updates
    """
    # Ensure output is saved in upload folder
    output_csv_path = ensure_upload_folder(output_csv_path)
    
    all_data = []
    total_pixels = target_size[0] * target_size[1]
    
    # Process each folder
    for folder_path in input_folders:
        folder_path = Path(folder_path)
        
        if not folder_path.is_dir():
            msg = f"Warning: {folder_path} is not a directory, skipping..."
            print(msg)
            if progress_callback:
                progress_callback(msg)
            continue
        
        # Extract label from folder name
        try:
            label = int(folder_path.name)
        except ValueError:
            msg = f"Warning: Folder name '{folder_path.name}' is not an integer, skipping..."
            print(msg)
            if progress_callback:
                progress_callback(msg)
            continue
        
        msg = f"Processing folder: {folder_path.name} (label: {label})"
        print(msg)
        if progress_callback:
            progress_callback(msg)
        
        # Find all .jpg files in the folder (case-insensitive to avoid duplicates on Windows)
        jpg_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.JPG'))
        # Remove duplicates (Windows filesystem is case-insensitive, so *.jpg and *.JPG match same files)
        jpg_files = list(set(jpg_files))
        
        if not jpg_files:
            msg = f"  No .jpg files found in {folder_path}"
            print(msg)
            if progress_callback:
                progress_callback(msg)
            continue
        
        # Process each image
        for img_path in jpg_files:
            msg = f"  Processing: {img_path.name}"
            print(msg)
            if progress_callback:
                progress_callback(msg)
            
            pixel_values = process_image(img_path, target_size)
            
            if pixel_values is not None:
                if len(pixel_values) != total_pixels:
                    msg = f"  Warning: Expected {total_pixels} pixels, got {len(pixel_values)}"
                    print(msg)
                    if progress_callback:
                        progress_callback(msg)
                    continue
                
                # Create row: label + pixel values
                row = [label] + pixel_values.tolist()
                all_data.append(row)
    
    if not all_data:
        msg = "No data to write. Please check your input folders."
        print(msg)
        if progress_callback:
            progress_callback(msg)
        return False
    
    # Write to CSV
    msg = f"\nWriting {len(all_data)} rows to {output_csv_path}..."
    print(msg)
    if progress_callback:
        progress_callback(msg)
    
    # Create column headers
    headers = ['label'] + [f'pixel{i}' for i in range(total_pixels)]
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(all_data)
    
    msg = f"Successfully created {output_csv_path}\nTotal images processed: {len(all_data)}\nImage size: {target_size[0]}x{target_size[1]} = {total_pixels} pixels per image"
    print(msg)
    if progress_callback:
        progress_callback(msg)
    
    return True


def main(args=None):
    """
    Main function to handle user input and process folders.
    
    Args:
        args: Optional argparse.Namespace object with command-line arguments.
              If None, will use interactive mode.
    """
    input_folders = []
    output_csv = "upload/dataset.csv"
    target_size = (128, 128)
    
    # If args provided, use command-line mode
    if args:
        output_csv = args.output
        target_size = tuple(args.size)
        
        # If parent-dir is specified, get all subdirectories
        if args.parent_dir:
            parent_path = Path(args.parent_dir)
            if not parent_path.exists() or not parent_path.is_dir():
                print(f"Error: {args.parent_dir} does not exist or is not a directory")
                return
            
            # Get all subdirectories
            input_folders = [d for d in parent_path.iterdir() if d.is_dir()]
            print(f"Found {len(input_folders)} subdirectories in {args.parent_dir}")
        
        # If folders are specified via command line
        elif args.folders:
            for folder in args.folders:
                folder_path = Path(folder)
                if folder_path.exists() and folder_path.is_dir():
                    input_folders.append(folder_path)
                else:
                    print(f"Warning: {folder} does not exist or is not a directory")
    
    # Interactive mode
    if not input_folders:
        print("="*70)
        print(" IMAGE TO CSV CONVERTER ".center(70))
        print("="*70)
        print("\nThis program converts .jpg images from labeled folders to CSV format.")
        print("Folder names should be integers representing the labels.")
        print("Images will be resized to 128x128 and converted to grayscale.\n")
        
        # Get input folders
        print("Enter folder paths (one per line, or press Enter twice to finish):")
        while True:
            folder = input().strip()
            if not folder:
                if input_folders:
                    break
                else:
                    print("Please enter at least one folder path.")
                    continue
            
            folder_path = Path(folder)
            if folder_path.exists() and folder_path.is_dir():
                input_folders.append(folder_path)
                print(f"  Added: {folder_path}")
            else:
                print(f"  Warning: {folder} does not exist or is not a directory")
        
        # Get output CSV path
        output_csv_input = input("\nEnter output CSV file path (default: upload/dataset.csv): ").strip()
        if output_csv_input:
            output_csv = output_csv_input
        else:
            output_csv = "upload/dataset.csv"
    
    if not input_folders:
        print("No valid folders provided. Exiting.")
        return
    
    # Process folders
    try:
        process_folders(input_folders, output_csv, target_size=target_size)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def methodology_to_script(methodology):
    """
    Map methodology string to script filename.
    
    Args:
        methodology: Methodology string (e.g., "Python_QPCA + Regression", "Quantum PCA + LinearRegression", "Matlab_SVDPCA + KNN")
    
    Returns:
        Script filename or None if not found
    """
    methodology_lower = methodology.lower()
    
    # Map different variations of methodology strings to script names
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


def create_image_to_csv_page(parent_frame, on_back=None, on_next=None, shared_state=None):
    """
    Create image to CSV converter page content in the given parent frame.
    
    Args:
        parent_frame: Parent frame to create the page in
        on_back: Optional callback function to call when "Back" button is clicked
        on_next: Optional callback function to call when "Next step" button is clicked
        shared_state: Optional shared state object to get methodology
    """
    if not GUI_AVAILABLE:
        error_label = ttk.Label(parent_frame, text="GUI not available. Please use command-line mode.")
        error_label.pack()
        return
    
    # Get root window for callbacks
    root = parent_frame.winfo_toplevel()
    
    # Variables
    selected_folders = []
    output_path = tk.StringVar(value="upload/dataset.csv")
    
    # Main frame
    main_frame = ttk.Frame(parent_frame, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, text="Image to CSV Converter", font=("Arial", 16, "bold"))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    
    # Instructions
    info_text = "Select labeled folders (folder names should be integers).\nEach folder should contain .jpg images of that label."
    info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT)
    info_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
    
    # Create two-column layout using PanedWindow
    paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
    paned.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    main_frame.rowconfigure(2, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    
    # Left column: Upload area
    left_frame = ttk.Frame(paned, padding="10")
    paned.add(left_frame, weight=1)
    
    # Right column: Status area
    right_frame = ttk.Frame(paned, padding="10")
    paned.add(right_frame, weight=1)
    
    # Folder selection section (in left column)
    folder_frame = ttk.LabelFrame(left_frame, text="Select Folders", padding="10")
    folder_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    folder_frame.columnconfigure(0, weight=1)
    folder_frame.rowconfigure(1, weight=1)
    
    # Upload button
    upload_btn = ttk.Button(folder_frame, text="📁 Upload Folder", command=lambda: select_folder())
    upload_btn.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
    
    # Folder listbox with scrollbar
    listbox_frame = ttk.Frame(folder_frame)
    listbox_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    listbox_frame.columnconfigure(0, weight=1)
    folder_frame.rowconfigure(1, weight=1)
    
    folder_listbox = tk.Listbox(listbox_frame, height=8)
    folder_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=folder_listbox.yview)
    scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    folder_listbox.configure(yscrollcommand=scrollbar.set)
    
    # Remove button
    remove_btn = ttk.Button(folder_frame, text="Remove Selected", command=lambda: remove_folder())
    remove_btn.grid(row=2, column=0, sticky=tk.W)
    
    def select_folder():
        """Open folder selection dialog."""
        folder = filedialog.askdirectory(title="Select a labeled folder")
        if folder:
            folder_path = Path(folder)
            if folder_path not in selected_folders:
                selected_folders.append(folder_path)
                folder_listbox.insert(tk.END, str(folder_path))
            else:
                messagebox.showinfo("Info", "Folder already selected.")
    
    def remove_folder():
        """Remove selected folder from list."""
        selection = folder_listbox.curselection()
        if selection:
            index = selection[0]
            folder_listbox.delete(index)
            selected_folders.pop(index)
    
    # Output file selection (in left column)
    output_frame = ttk.LabelFrame(left_frame, text="Output CSV File", padding="10")
    output_frame.pack(fill=tk.X, pady=(0, 10))
    output_frame.columnconfigure(1, weight=1)
    
    ttk.Label(output_frame, text="Output:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
    output_entry = ttk.Entry(output_frame, textvariable=output_path, width=40)
    output_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
    
    def select_output():
        """Open file save dialog for output CSV."""
        # Set initial directory to upload folder
        initial_dir = Path("upload")
        initial_dir.mkdir(exist_ok=True)
        
        filename = filedialog.asksaveasfilename(
            title="Save CSV As",
            initialdir=str(initial_dir),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            # Always save in upload folder, extract just the filename
            filename_path = Path(filename)
            output_path.set(f"upload/{filename_path.name}")
    
    output_btn = ttk.Button(output_frame, text="Browse...", command=select_output)
    output_btn.grid(row=0, column=2)
    
    # Progress/Status area (in right column)
    status_frame = ttk.LabelFrame(right_frame, text="Status & Messages", padding="10")
    status_frame.pack(fill=tk.BOTH, expand=True)
    status_frame.columnconfigure(0, weight=1)
    status_frame.rowconfigure(0, weight=1)
    
    status_text = scrolledtext.ScrolledText(status_frame, height=30, width=60, wrap=tk.WORD)
    status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def update_status(message):
        """Update status text area."""
        status_text.insert(tk.END, message + "\n")
        status_text.see(tk.END)
        root.update_idletasks()
    
    # Process button
    def process_images():
        """Process selected folders and create CSV."""
        if not selected_folders:
            messagebox.showwarning("Warning", "Please select at least one folder.")
            return
        
        if not output_path.get():
            messagebox.showwarning("Warning", "Please specify an output CSV file path.")
            return
        
        # Disable process button during processing
        process_btn.config(state=tk.DISABLED)
        upload_btn.config(state=tk.DISABLED)
        status_text.delete(1.0, tk.END)
        
        def process_thread():
            try:
                success = process_folders(
                    selected_folders,
                    output_path.get(),
                    target_size=(128, 128),
                    progress_callback=update_status
                )
                if success:
                    root.after(0, lambda: messagebox.showinfo("Success", f"CSV file created successfully!\n{output_path.get()}"))
                else:
                    root.after(0, lambda: messagebox.showerror("Error", "Failed to create CSV file. Check status for details."))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                update_status(error_msg)
                root.after(0, lambda: messagebox.showerror("Error", error_msg))
            finally:
                root.after(0, lambda: process_btn.config(state=tk.NORMAL))
                root.after(0, lambda: upload_btn.config(state=tk.NORMAL))
        
        # Run processing in separate thread to keep GUI responsive
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()
    
    # Note: Docker execution function has been moved to show_script_and_execute.py
    
    # Button frame for actions (in left column)
    button_frame = ttk.Frame(left_frame)
    button_frame.pack(fill=tk.X, pady=(10, 0))
    button_frame.columnconfigure(1, weight=1)
    
    # Back button
    if on_back:
        def go_back():
            """Navigate back to previous step"""
            on_back()
        
        back_btn = ttk.Button(button_frame, text="⬅️ Back", command=go_back)
        back_btn.grid(row=0, column=0, sticky=tk.W)
    
    # Process button
    process_btn = ttk.Button(button_frame, text="🚀 Process Images", command=process_images)
    process_btn.grid(row=0, column=1, padx=5)
    
    # Generate script button (navigate to script page)
    if shared_state and shared_state.methodology:
        methodology_display = shared_state.methodology
    else:
        methodology_display = "No methodology selected"
    
    def generate_script():
        """Navigate to script generation page and copy script to temporary folder"""
        if not shared_state or not shared_state.methodology:
            messagebox.showwarning("Warning", "No methodology selected. Please go back to page 1 and predict a methodology first.")
            return
        
        # Check if CSV file exists in upload folder
        csv_file = Path("upload") / output_path.get().split("/")[-1] if "/" in output_path.get() else Path("upload") / output_path.get()
        if not csv_file.exists():
            messagebox.showwarning("Warning", f"CSV file not found: {csv_file}\n\nPlease process images first to create the CSV file.")
            return
        
        # Map methodology to script name
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
        
        script_name = map_methodology_to_script(shared_state.methodology)
        if not script_name:
            messagebox.showerror("Error", f"Could not map methodology '{shared_state.methodology}' to a script.\n\nSupported methodologies:\n- Python_QPCA + Regression\n- QPCA (python) + CNN (python)\n- Python_QPCA + XGBoost\n- Python_QFM + Regression\n- Python_QFM + CNN\n- Python_QFM + XGBoost\n- PQK(python, TensorFlow Quantum libraries) + CNN (python, TensorFlow libraries)\n- Matlab_SVDPCA + KNN")
            return
        
        # Create temporary folder if it doesn't exist
        temp_folder = Path("temporary_model_training_script")
        temp_folder.mkdir(exist_ok=True)
        
        # Copy script from universal_training_script to temporary folder
        source_script = Path("universal_training_script") / script_name
        dest_script = temp_folder / script_name
        
        if not source_script.exists():
            messagebox.showerror("Error", f"Source script not found: {source_script}")
            return
        
        try:
            import shutil
            shutil.copy2(source_script, dest_script)
            print(f"Copied script from {source_script} to {dest_script}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy script: {str(e)}")
            return
        
        # Store script name in shared state for use in next page
        shared_state.script_name = script_name
        
        # Navigate to next page (script page)
        if on_next:
            on_next()
    
    generate_btn = ttk.Button(
        button_frame, 
        text="📝 Start Generate Model Training Script", 
        command=generate_script
    )
    generate_btn.grid(row=0, column=2, sticky=tk.E)
    
    # Show methodology info (in left column)
    if shared_state and shared_state.methodology:
        info_label = ttk.Label(
            left_frame, 
            text=f"Selected Methodology: {shared_state.methodology}",
            font=("Arial", 10, "italic"),
            foreground="blue"
        )
        info_label.pack(pady=(10, 0))


def create_gui():
    """
    Create a standalone GUI application with upload button for folder selection.
    """
    if not GUI_AVAILABLE:
        print("GUI not available. Please use command-line mode.")
        return
    
    root = tk.Tk()
    root.title("Image to CSV Converter")
    root.geometry("700x600")
    
    create_image_to_csv_page(root)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert .jpg images from labeled folders to CSV format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GUI mode (default)
  python image_to_csv.py
  
  # GUI mode explicitly
  python image_to_csv.py --gui
  
  # Command line mode
    python image_to_csv.py --folders folder1 folder2 folder3 --output dataset.csv
  
  # Process folders from a parent directory
    python image_to_csv.py --parent-dir /path/to/parent --output dataset.csv
        """
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch GUI interface (default if no other arguments provided)'
    )
    parser.add_argument(
        '--folders',
        nargs='+',
        help='List of folder paths (folder names should be integer labels)'
    )
    parser.add_argument(
        '--parent-dir',
        type=str,
        help='Parent directory containing labeled subfolders (all subfolders will be processed)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='upload/dataset.csv',
        help='Output CSV file path (default: upload/dataset.csv, will be saved in upload folder)'
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=[128, 128],
        metavar=('WIDTH', 'HEIGHT'),
        help='Target image size in pixels (default: 128 128)'
    )
    
    args = parser.parse_args()
    
    # If GUI is requested or no command-line arguments provided, launch GUI
    if args.gui or (not args.folders and not args.parent_dir):
        if GUI_AVAILABLE:
            create_gui()
        else:
            print("GUI not available. Falling back to interactive mode.")
            main(None)
    else:
        # Command-line mode
        main(args)

