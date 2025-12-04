# -*- coding: utf-8 -*-
"""
LLM Chat Widget
Provides a GUI widget for LLM-assisted script modification.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
import threading
import requests
import re


def load_license(license_type="OPENROUTER_API_KEY"):
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
                first_line = None  # For backward compatibility
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
                    # Store first non-comment line for backward compatibility
                    elif first_line is None:
                        first_line = line
                
                # Fallback: if looking for OPENROUTER_API_KEY and no key=value format found,
                # return first non-comment line (backward compatibility)
                if license_type == "OPENROUTER_API_KEY" and first_line:
                    return first_line
        return None
    except Exception as e:
        print(f"Error loading license: {e}")
        return None


def create_llm_chat_widget(parent_frame, script_text_widget, shared_state, root, version_callback=None):
    """
    Create LLM chat widget for script modification.
    
    Args:
        parent_frame: Parent frame to create the widget in
        script_text_widget: Text widget displaying the script (will be updated)
        shared_state: Shared state object containing script_name
        root: Root window for callbacks
        version_callback: Optional callback function to call when script is modified
    
    Returns:
        tuple: (llm_frame, llm_chat_text, llm_input_entry, llm_send_btn)
    """
    # LLM Chat area
    llm_frame = ttk.LabelFrame(parent_frame, text="🤖 LLM Assistant (Modify Script)", padding="10")
    llm_frame.pack(fill=tk.BOTH, expand=True)
    llm_frame.columnconfigure(0, weight=1)
    llm_frame.rowconfigure(0, weight=1)
    
    # LLM chat output area
    llm_chat_text = scrolledtext.ScrolledText(llm_frame, height=8, width=60, wrap=tk.WORD, font=("Arial", 9))
    llm_chat_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
    llm_chat_text.insert(1.0, "Do you satisfied with current script's performance? \nYou can chat with LLM to modify your script. Ask for changes and the LLM will apply them.\n\n")
    llm_chat_text.config(state=tk.DISABLED)
    
    # LLM input area
    llm_input_frame = ttk.Frame(llm_frame)
    llm_input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
    llm_input_frame.columnconfigure(0, weight=1)
    
    llm_input_entry = tk.Text(llm_input_frame, height=3, wrap=tk.WORD, font=("Arial", 9))
    llm_input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
    
    def send_to_llm():
        """Send user message to LLM and apply changes to script"""
        user_message = llm_input_entry.get(1.0, tk.END).strip()
        if not user_message:
            return
        
        if not shared_state or not shared_state.script_name:
            messagebox.showwarning("Warning", "No script selected. Please generate a script first.")
            return
        
        script_path = Path("temporary_model_training_script") / shared_state.script_name
        if not script_path.exists():
            messagebox.showwarning("Warning", f"Script not found: {script_path}")
            return
        
        # Disable input during processing
        llm_send_btn.config(state=tk.DISABLED)
        llm_input_entry.config(state=tk.DISABLED)
        
        def llm_thread():
            try:
                # Get script_text_widget from frame (may be updated dynamically)
                # Check frame attribute first, then fall back to parameter
                current_script_widget = getattr(llm_frame, 'script_text_widget', None)
                if current_script_widget is None:
                    current_script_widget = script_text_widget
                
                # Load current script
                with open(script_path, 'r', encoding='utf-8') as f:
                    current_script = f.read()
                
                # Update chat with user message
                llm_chat_text.config(state=tk.NORMAL)
                llm_chat_text.insert(tk.END, f"👤 You: {user_message}\n\n")
                llm_chat_text.insert(tk.END, "🤖 LLM: Processing...\n\n")
                llm_chat_text.see(tk.END)
                llm_chat_text.config(state=tk.DISABLED)
                root.update_idletasks()
                
                # Load license/API key
                api_key = load_license("OPENROUTER_API_KEY")
                if not api_key:
                    error_msg = "License/API key not found. Please create license.txt with your OpenRouter API key."
                    llm_chat_text.config(state=tk.NORMAL)
                    llm_chat_text.insert(tk.END, f"❌ Error: {error_msg}\n\n")
                    llm_chat_text.see(tk.END)
                    llm_chat_text.config(state=tk.DISABLED)
                    root.after(0, lambda: messagebox.showerror("Error", error_msg))
                    return
                
                # Prepare prompt for LLM
#                 system_prompt = """You are a helpful code assistant. The user wants to modify a Python training script.

# IMPORTANT INSTRUCTIONS:
# 1. Make MINIMAL, NON-AGGRESSIVE changes - only modify what the user specifically asks for
# 2. Preserve all existing functionality unless explicitly asked to change it
# 3. Return ONLY the modified Python code, wrapped in ```python code blocks
# 4. If the user's request is unclear or risky, explain why and ask for clarification
# 5. Maintain code style, comments, and structure
# 6. Do NOT add unnecessary imports or changes
# 7. If you cannot safely make the change, explain why instead of making risky modifications

# The user will provide:
# - The current script code
# - Their modification request

# You should return the complete modified script with only the necessary changes applied."""
                system_prompt = """You are a helpful code assistant. The user wants to modify a Python training script.

IMPORTANT INSTRUCTIONS:
1. Preserve all existing functionality unless explicitly asked to change it
2. Return ONLY the modified code, wrapped in ```python code blocks
3. If the user's request is unclear or risky, explain why and ask for clarification
4. Maintain code style, comments, and structure
5. Do NOT add unnecessary imports or changes
6. If you cannot safely make the change, explain why instead of making risky modifications

The user will provide:
- The current script code
- Their modification request

You should return the complete modified script with only the necessary changes applied."""
                user_prompt = f"""Current script:

```python
{current_script}
```

User request: {user_message}

Please modify the script according to the user's request. Return the COMPLETE MODIFIED SCRIPT wrapped in ```python code blocks."""
                
                # Call OpenRouter API
                api_url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "openai/gpt-4o-mini",  # You can change this model
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3  # Lower temperature for more consistent, non-aggressive changes
                }
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                llm_response = result['choices'][0]['message']['content']
                
                # Extract code from markdown code blocks
                code_pattern = r'```python\s*(.*?)```'
                matches = re.findall(code_pattern, llm_response, re.DOTALL)
                
                if not matches:
                    # Try without language specifier
                    code_pattern = r'```\s*(.*?)```'
                    matches = re.findall(code_pattern, llm_response, re.DOTALL)
                
                if matches:
                    modified_script = matches[0].strip()
                    
                    # Read current script content to save as old version
                    old_content = None
                    if script_path.exists():
                        with open(script_path, 'r', encoding='utf-8') as f:
                            old_content = f.read()
                    
                    # Only proceed if content actually changed
                    if old_content == modified_script:
                        llm_chat_text.config(state=tk.NORMAL)
                        llm_chat_text.insert(tk.END, "⚠️ No changes detected in the script.\n\n")
                        llm_chat_text.see(tk.END)
                        llm_chat_text.config(state=tk.DISABLED)
                        root.after(0, lambda: llm_send_btn.config(state=tk.NORMAL))
                        root.after(0, lambda: llm_input_entry.config(state=tk.NORMAL))
                        return
                    
                    # Save modified script first
                    with open(script_path, 'w', encoding='utf-8') as f:
                        f.write(modified_script)
                    
                    # Increment version if callback provided
                    # Check both the parameter and the frame attribute (in case it was updated)
                    version_cb = getattr(llm_frame, 'version_callback', None)
                    if version_cb is None:
                        version_cb = version_callback
                    
                    if version_cb:
                        # Call version callback in the main thread
                        # Use popup window if available, otherwise use root
                        # This ensures the callback runs in the popup's event loop
                        callback_func = version_cb
                        popup_win = getattr(llm_frame, 'popup_window', None)
                        target_window = popup_win if popup_win else root
                        
                        print(f"DEBUG: LLM - Calling version callback: {callback_func}")
                        print(f"DEBUG: LLM - Using window: {target_window}")
                        print(f"DEBUG: LLM - Capturing user prompt: {user_message[:100] if user_message else 'None'}...")
                        
                        # CRITICAL: Capture user_message in a local variable BEFORE defining the nested function
                        # This ensures the closure captures the correct value
                        # Store it in a variable that will be accessible in the nested function
                        captured_user_message = str(user_message) if user_message else None  # Explicitly convert to string and capture
                        print(f"DEBUG: LLM - Captured prompt for saving (type: {type(captured_user_message)}): {captured_user_message[:100] if captured_user_message else 'None'}...")
                        
                        # Also capture modified_script to ensure it's available
                        captured_modified_script = modified_script
                        
                        def increment_and_save():
                            try:
                                print(f"DEBUG: LLM - increment_and_save() callback executed")
                                print(f"DEBUG: LLM - captured_user_message available: {captured_user_message is not None}")
                                print(f"DEBUG: LLM - captured_user_message value: {captured_user_message[:50] if captured_user_message else 'None'}...")
                                
                                # Get current version BEFORE increment
                                popup_win = getattr(llm_frame, 'popup_window', None)
                                print(f"DEBUG: LLM - popup_win available: {popup_win is not None}")
                                
                                if popup_win:
                                    current_version = getattr(popup_win, 'script_version', 1)
                                    print(f"DEBUG: LLM - Current version before increment: {current_version}")
                                    
                                    # Increment version first
                                    callback_func()
                                    
                                    # After version is incremented, save the NEW version only
                                    # The old version should already be saved (either from initial copy or pre_execution)
                                    new_version = getattr(popup_win, 'script_version', None)
                                    print(f"DEBUG: LLM - New version after increment: {new_version}")
                                    
                                    if new_version and shared_state and shared_state.script_name:
                                        try:
                                            from log_history_helper_functions.version_history import VersionHistoryManager
                                            version_manager = VersionHistoryManager()
                                            
                                            # Use captured user message - this is the prompt the user typed into the LLM input field
                                            # Double-check that we have the prompt
                                            prompt_to_save = captured_user_message if captured_user_message else None
                                            
                                            # Debug output to verify prompt is available
                                            print(f"DEBUG: LLM - About to save version {new_version}")
                                            print(f"DEBUG: LLM - Prompt to save (type: {type(prompt_to_save)}): {prompt_to_save[:100] if prompt_to_save else 'None'}...")
                                            print(f"DEBUG: LLM - Prompt length: {len(prompt_to_save) if prompt_to_save else 0}")
                                            
                                            # STEP 1: Save the version (with or without prompt)
                                            print(f"DEBUG: LLM - Saving version {new_version} with prompt using save_version")
                                            saved_new_version = version_manager.save_version(
                                                shared_state.script_name, 
                                                captured_modified_script, 
                                                "llm_modification", 
                                                new_version,
                                                user_prompt=prompt_to_save  # Store user's input prompt
                                            )
                                            print(f"DEBUG: LLM - Saved version as v{saved_new_version} to history")
                                            
                                            # STEP 2: Use dedicated method to ensure prompt is saved and protected
                                            if prompt_to_save:
                                                print(f"DEBUG: LLM - Using dedicated update_user_prompt method to save: {prompt_to_save[:50]}...")
                                                success = version_manager.update_user_prompt(
                                                    shared_state.script_name,
                                                    saved_new_version,
                                                    prompt_to_save
                                                )
                                                if success:
                                                    print(f"DEBUG: LLM - Prompt saved successfully using dedicated method")
                                                else:
                                                    print(f"ERROR: LLM - Failed to save prompt using dedicated method")
                                            
                                            # STEP 3: Verify the prompt was actually saved by reloading from disk
                                            verify_manager = VersionHistoryManager()
                                            saved_versions = verify_manager.get_all_versions(shared_state.script_name)
                                            saved_version = next((v for v in saved_versions if v["version"] == saved_new_version), None)
                                            if saved_version:
                                                saved_prompt = saved_version.get("user_prompt")
                                                print(f"DEBUG: LLM - Verified saved prompt in metadata: {saved_prompt[:50] if saved_prompt else 'None'}...")
                                                if saved_prompt != prompt_to_save:
                                                    print(f"ERROR: Prompt mismatch! Expected: {prompt_to_save[:50] if prompt_to_save else 'None'}, Got: {saved_prompt[:50] if saved_prompt else 'None'}")
                                                    # Final attempt: use dedicated method again
                                                    print(f"DEBUG: LLM - Final attempt to save prompt using dedicated method...")
                                                    version_manager.update_user_prompt(
                                                        shared_state.script_name,
                                                        saved_new_version,
                                                        prompt_to_save
                                                    )
                                            else:
                                                print(f"ERROR: Could not find saved version {saved_new_version} in metadata after saving!")
                                        except Exception as e:
                                            print(f"ERROR: Could not save new version to history: {e}")
                                            import traceback
                                            traceback.print_exc()
                                    else:
                                        print(f"DEBUG: LLM - Cannot save version: new_version={new_version}, shared_state={shared_state is not None}, script_name={shared_state.script_name if shared_state else None}")
                                else:
                                    print(f"DEBUG: LLM - popup_win is None, cannot increment version")
                                    print(f"DEBUG: LLM - Attempting fallback: save version without increment")
                                    # Fallback: save version without incrementing if popup is not available
                                    if shared_state and shared_state.script_name:
                                        try:
                                            from log_history_helper_functions.version_history import VersionHistoryManager
                                            version_manager = VersionHistoryManager()
                                            
                                            # Get current latest version and increment it manually
                                            existing_versions = version_manager.get_all_versions(shared_state.script_name)
                                            if existing_versions:
                                                latest_version = max(v["version"] for v in existing_versions)
                                                new_version = latest_version + 1
                                            else:
                                                new_version = 1
                                            
                                            print(f"DEBUG: LLM - Fallback: Saving as version {new_version} with prompt: {captured_user_message[:50] if captured_user_message else 'None'}...")
                                            saved_version = version_manager.save_version(
                                                shared_state.script_name,
                                                captured_modified_script,
                                                "llm_modification",
                                                version=new_version,
                                                user_prompt=captured_user_message if captured_user_message else None
                                            )
                                            print(f"DEBUG: LLM - Fallback: Saved version {saved_version} with prompt: {captured_user_message[:50] if captured_user_message else 'None'}...")
                                            
                                            # Verify
                                            saved_versions = version_manager.get_all_versions(shared_state.script_name)
                                            saved_version_entry = next((v for v in saved_versions if v["version"] == saved_version), None)
                                            if saved_version_entry:
                                                saved_prompt = saved_version_entry.get("user_prompt")
                                                print(f"DEBUG: LLM - Fallback: Verified saved prompt: {saved_prompt[:50] if saved_prompt else 'None'}...")
                                        except Exception as e:
                                            print(f"ERROR: Could not save version as fallback: {e}")
                                            import traceback
                                            traceback.print_exc()
                                    else:
                                        print(f"DEBUG: LLM - Fallback failed: shared_state={shared_state is not None}, script_name={shared_state.script_name if shared_state else None}")
                            except Exception as e:
                                print(f"ERROR: Error in increment_and_save: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        try:
                            target_window.after(0, increment_and_save)
                        except Exception as e:
                            print(f"DEBUG: Error scheduling version callback: {e}")
                            # Try to call directly as fallback
                            try:
                                increment_and_save()
                            except Exception as e2:
                                print(f"DEBUG: Error calling version callback directly: {e2}")
                    else:
                        # No version callback - just save the new version without incrementing
                        if shared_state and shared_state.script_name:
                            try:
                                from user_interface.version_history import VersionHistoryManager
                                version_manager = VersionHistoryManager()
                                version_manager.save_version(
                                    shared_state.script_name, 
                                    modified_script, 
                                    "llm_modification"
                                )
                            except Exception as e:
                                print(f"Warning: Could not save version to history: {e}")
                    
                    # Update script display (use current_script_widget from closure)
                    # Try to get widget reference again in case it was set after thread started
                    # This ensures we have the latest reference even if popup opened during LLM processing
                    current_script_widget = getattr(llm_frame, 'script_text_widget', None)
                    if current_script_widget is None:
                        current_script_widget = script_text_widget  # Fall back to original parameter
                    
                    if current_script_widget is not None:
                        # Capture widget reference in lambda defaults to avoid closure issues
                        widget = current_script_widget
                        script_content = modified_script
                        root.after(0, lambda w=widget: w.config(state=tk.NORMAL) if w else None)
                        root.after(0, lambda w=widget: w.delete(1.0, tk.END) if w else None)
                        root.after(0, lambda w=widget, s=script_content: w.insert(1.0, s) if w else None)
                        root.after(0, lambda w=widget: w.config(state=tk.NORMAL) if w else None)
                    else:
                        # Widget not available, use refresh callback as fallback
                        refresh_callback = getattr(llm_frame, 'refresh_script_callback', None)
                        if refresh_callback:
                            root.after(0, refresh_callback)
                        else:
                            # Log warning if neither widget nor callback is available
                            llm_chat_text.config(state=tk.NORMAL)
                            llm_chat_text.insert(tk.END, "⚠️ Warning: Script editor not connected. Please open the Python editor popup.\n\n")
                            llm_chat_text.see(tk.END)
                            llm_chat_text.config(state=tk.DISABLED)
                    
                    # Also refresh script display if callback is available (reloads from file)
                    # This ensures the display is in sync even if direct update worked
                    refresh_callback = getattr(llm_frame, 'refresh_script_callback', None)
                    if refresh_callback and current_script_widget is not None:
                        # Only refresh if we also updated directly (to avoid double update)
                        root.after(100, refresh_callback)  # Small delay to let direct update complete
                    
                    # Update chat
                    llm_chat_text.config(state=tk.NORMAL)
                    llm_chat_text.delete("end-2l", "end-1l")  # Remove "Processing..."
                    llm_chat_text.insert(tk.END, f"✅ Script modified successfully!\n\n")
                    llm_chat_text.see(tk.END)
                    llm_chat_text.config(state=tk.DISABLED)
                else:
                    # No code block found, show explanation
                    llm_chat_text.config(state=tk.NORMAL)
                    llm_chat_text.delete("end-2l", "end-1l")  # Remove "Processing..."
                    llm_chat_text.insert(tk.END, f"💬 {llm_response}\n\n")
                    llm_chat_text.see(tk.END)
                    llm_chat_text.config(state=tk.DISABLED)
                
            except requests.exceptions.RequestException as e:
                error_msg = f"API Error: {str(e)}"
                llm_chat_text.config(state=tk.NORMAL)
                llm_chat_text.delete("end-2l", "end-1l")  # Remove "Processing..."
                llm_chat_text.insert(tk.END, f"❌ {error_msg}\n\n")
                llm_chat_text.see(tk.END)
                llm_chat_text.config(state=tk.DISABLED)
                root.after(0, lambda: messagebox.showerror("Error", error_msg))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                llm_chat_text.config(state=tk.NORMAL)
                llm_chat_text.delete("end-2l", "end-1l")  # Remove "Processing..."
                llm_chat_text.insert(tk.END, f"❌ {error_msg}\n\n")
                llm_chat_text.see(tk.END)
                llm_chat_text.config(state=tk.DISABLED)
                root.after(0, lambda: messagebox.showerror("Error", error_msg))
            finally:
                root.after(0, lambda: llm_send_btn.config(state=tk.NORMAL))
                root.after(0, lambda: llm_input_entry.config(state=tk.NORMAL))
        
        # Clear input before starting thread
        llm_input_entry.delete(1.0, tk.END)
        
        # Run in separate thread
        thread = threading.Thread(target=llm_thread, daemon=True)
        thread.start()
    
    llm_send_btn = ttk.Button(llm_input_frame, text="Send", command=send_to_llm)
    llm_send_btn.grid(row=0, column=1, sticky=tk.E)
    
    # Bind Enter key to send (Ctrl+Enter for newline)
    def on_llm_input_key(event):
        if event.state & 0x4 and event.keysym == 'Return':  # Ctrl+Enter
            send_to_llm()
            return "break"
    
    llm_input_entry.bind('<KeyPress>', on_llm_input_key)
    
    return llm_frame, llm_chat_text, llm_input_entry, llm_send_btn

