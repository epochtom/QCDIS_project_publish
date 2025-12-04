# -*- coding: utf-8 -*-
"""
Prompt Predictor GUI
Allows users to input prompts and get neural network predictions for recommended methodologies.
"""

import os
import json
import torch
import pickle
import threading
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

try:
    import tkinter as tk
    from tkinter import messagebox, ttk, scrolledtext
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Disable TensorFlow warnings and set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


# Neural Network Model (same as train_neural_network.py)
class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=50, hidden_dim=100, num_classes=5):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(self.dropout(hidden[-1]))
        return output


def simple_tokenize(text, vocab, max_length=20):
    """Convert text to numbers using provided vocab"""
    words = text.lower().split()
    token_ids = [vocab.get(word, 0) for word in words]
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids.extend([0] * (max_length - len(token_ids)))
    return torch.tensor(token_ids, dtype=torch.long)


def load_model():
    """Load the trained model, vocabulary, and label encoder"""
    model_folder = Path("neural_network_model")
    
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder '{model_folder}' not found. Please train the model first.")
    
    model_path = model_folder / 'model.pth'
    label_encoder_path = model_folder / 'label_encoder.pkl'
    vocab_path = model_folder / 'vocab.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder file '{label_encoder_path}' not found.")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file '{vocab_path}' not found.")
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Load label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Determine number of classes
    num_classes = len(label_encoder.classes_)
    
    # Create and load model
    model = SimpleTextClassifier(vocab_size=1000, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, vocab, label_encoder


def predict_model_type(prompt, model, vocab, label_encoder):
    """Predict model type from prompt"""
    model.eval()
    with torch.no_grad():
        tokenized = simple_tokenize(prompt, vocab).unsqueeze(0)
        output = model(tokenized)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label


def create_prompt_predictor_page(parent_frame, on_next=None, shared_state=None):
    """Create prompt predictor page content in the given parent frame
    
    Args:
        parent_frame: Parent frame to create the page in
        on_next: Optional callback function to call when "Next step" button is clicked
        shared_state: Optional shared state object to store methodology
    """
    if not GUI_AVAILABLE:
        error_label = ttk.Label(parent_frame, text="GUI not available. Please install tkinter.")
        error_label.pack()
        return
    
    # Try to load model
    try:
        model, vocab, label_encoder = load_model()
    except Exception as e:
        error_label = ttk.Label(parent_frame, text=f"Error loading model: {e}\n\nPlease train the model first using train_neural_network.py", 
                                foreground="red", wraplength=600)
        error_label.pack(pady=20)
        return
    
    # Get root window for callbacks
    root = parent_frame.winfo_toplevel()
    
    # Main frame
    main_frame = ttk.Frame(parent_frame, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.columnconfigure(1, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="QCDIS", font=("Arial", 16, "bold"))
    title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
    
    # Instructions
    info_text = " Enter your problem below to get recommendations on: \n 1) HQC decomposition 2) platforms used"
    info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT, wraplength=650)
    info_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
    
    # Prompt input section
    prompt_frame = ttk.LabelFrame(main_frame, text="Enter Your Prompt", padding="10")
    prompt_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
    prompt_frame.columnconfigure(0, weight=1)
    
    prompt_text = scrolledtext.ScrolledText(prompt_frame, height=6, width=70, wrap=tk.WORD)
    prompt_text.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
    prompt_text.insert("1.0", "Classify the provided fashion	product images. These products spans accross 10 categories.")
    
    # Predict button
    def predict():
        """Predict methodology from prompt"""
        prompt = prompt_text.get("1.0", tk.END).strip()
        
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt.")
            return
        
        # Disable button during prediction
        predict_btn.config(state=tk.DISABLED)
        status_text.delete(1.0, tk.END)
        status_text.insert(tk.END, f"Processing prompt: {prompt}\n")
        root.update_idletasks()
        
        def predict_thread():
            try:
                predicted = predict_model_type(prompt, main_frame.model, main_frame.vocab, main_frame.label_encoder)
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, predicted)
                status_text.insert(tk.END, f"✓ Prediction complete!\n")
                status_text.insert(tk.END, f"Recommended methodology: {predicted}\n")
                
                # Store in shared state if available
                if shared_state:
                    shared_state.methodology = predicted
                    print(f"DEBUG: Stored methodology '{predicted}' in shared_state")
                else:
                    print("DEBUG: shared_state is None in prompt_predictor_gui")
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                status_text.insert(tk.END, error_msg + "\n")
                messagebox.showerror("Error", error_msg)
            finally:
                root.after(0, lambda: predict_btn.config(state=tk.NORMAL))
        
        # Run prediction in separate thread to keep GUI responsive
        thread = threading.Thread(target=predict_thread, daemon=True)
        thread.start()
    
    predict_btn = ttk.Button(prompt_frame, text="🔮 Predict Methodology", command=predict)
    predict_btn.grid(row=1, column=0, pady=(0, 0))
    
    # Result section
    result_frame = ttk.LabelFrame(main_frame, text="Recommended Methodology", padding="10")
    result_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    result_frame.columnconfigure(0, weight=1)
    result_frame.rowconfigure(0, weight=1)
    main_frame.rowconfigure(3, weight=1)
    
    result_text = scrolledtext.ScrolledText(result_frame, height=4, width=70, wrap=tk.WORD, 
                                            font=("Arial", 12, "bold"), bg="#f0f0f0")
    result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Status section
    status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
    status_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
    status_frame.columnconfigure(0, weight=1)
    status_frame.rowconfigure(0, weight=1)
    main_frame.rowconfigure(4, weight=1)
    
    status_text = scrolledtext.ScrolledText(status_frame, height=6, width=70)
    status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    status_text.insert(tk.END, "\n")
    
    # Clear button
    def clear_all():
        """Clear all inputs and results"""
        prompt_text.delete("1.0", tk.END)
        result_text.delete("1.0", tk.END)
        status_text.delete("1.0", tk.END)
        status_text.insert(tk.END, "Cleared. Ready for new input.\n")
    
    # Button frame for actions
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=5, column=0, columnspan=3, pady=(10, 0))
    
    clear_btn = ttk.Button(button_frame, text="🗑️ Clear All", command=clear_all)
    clear_btn.pack(side=tk.LEFT, padx=(0, 10))
    
    # Next step button
    if on_next:
        def next_step():
            """Navigate to next step"""
            on_next()
        
        next_btn = ttk.Button(button_frame, text="➡️ Next Step: Upload Dataset", command=next_step)
        next_btn.pack(side=tk.RIGHT)
    
    # Store model, vocab, label_encoder in the frame for access in predict function
    main_frame.model = model
    main_frame.vocab = vocab
    main_frame.label_encoder = label_encoder


def create_gui():
    """Create standalone GUI application for prompt prediction"""
    if not GUI_AVAILABLE:
        print("GUI not available. Please install tkinter.")
        return
    
    root = tk.Tk()
    root.title("Prompt Predictor - Neural Network Model Recommender")
    root.geometry("700x500")
    
    create_prompt_predictor_page(root)
    root.mainloop()


if __name__ == "__main__":
    create_gui()

