# -*- coding: utf-8 -*-
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import pennylane as qml

# -------------------------- Torch --------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ------------------- Plotting Imports -------------------
import matplotlib.pyplot as plt

# ============================= CONFIG =============================
class CFG:
    DATASET_PATH     = '/app/upload/dataset.csv'      # your CSV
    TARGET_COLUMN    = None                     # auto-detect
    TEST_SIZE        = 0.2
    RANDOM_STATE     = 42
    N_QUBITS         = 4                       # any integer >= 1
    BATCH_SIZE       = 32
    EPOCHS           = 20
    LR               = 1e-3
    JITTER           = 1e-6
    FORCE_CLASSICAL  = False                    # skip quantum → raw scaled data

    # ---------------------- 1-D CNN HYPERPARAMETERS ----------------------
    CONV_LAYERS      = [                        # list of (out_channels, kernel_size)
        (32, 3),
        (64, 3),
    ]
    ACTIVATION       = "relu"                   # "relu", "leaky_relu", "gelu", "tanh", "sigmoid"
    POOLING          = True                     # apply MaxPool after each conv
    POOL_SIZE        = 2
    DROPOUT          = 0.3
    DENSE_LAYERS     = [128, 64]                # hidden dense units (before final softmax)
    
    path_saving_plot = "/app/output"  # Directory for saving plots
    performance_plot_name = "performance_summary.png"  # Name of performance plot
    PLOT_SAVE_PATH = os.path.join(path_saving_plot, performance_plot_name)  # Full path
    # -----------------------------------------------------------------
# ==================================================================


def find_target_column(df):
    if 'label' in df.columns:
        return 'label'
    if df.iloc[:, 0].nunique() <= 20 and df.iloc[:, 0].dtype in ['int64', 'int32']:
        return df.columns[0]
    return df.columns[-1]


# ========================= QUANTUM FEATURE MAP =========================
def quantum_feature_map(X: np.ndarray, n_qubits: int) -> np.ndarray:
    n_samples, n_features = X.shape
    if n_features < n_qubits:
        pad = np.zeros((n_samples, n_qubits - n_features))
        X = np.hstack([X, pad])
    else:
        X = X[:, :n_qubits]

    X_norm = np.pi * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
    X_norm += np.random.normal(0, CFG.JITTER, X_norm.shape)

    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev, interface="numpy")
    def circuit(x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    phi = np.stack([circuit(x) for x in X_norm])
    return phi  # (n_samples, n_qubits)


def safe_quantum_mapping(X, n_qubits):
    try:
        print(f"  → Quantum Feature Map ({n_qubits} qubits)...")
        return quantum_feature_map(X, n_qubits)
    except Exception as e:
        print(f"  → Quantum failed ({e}), using scaled data")
        return X
# ======================================================================


# ========================= 1-D CNN MODEL =========================
class QuantumCNN(nn.Module):
    def __init__(self, n_qubits: int, n_classes: int):
        super().__init__()
        self.n_qubits = n_qubits

        # ---------- Build convolutional stack (1-D only) ----------
        layers = []
        in_ch = 1

        for out_ch, k in CFG.CONV_LAYERS:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2))
            layers.append(self._act())
            if CFG.POOLING:
                layers.append(nn.MaxPool1d(CFG.POOL_SIZE))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)

        # ---------- Compute flattened size ----------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_qubits)          # (B, C, L)
            out = self.conv(dummy)
            flat_size = out.view(1, -1).size(1)

        # ---------- Dense head ----------
        dense = []
        prev = flat_size
        for units in CFG.DENSE_LAYERS:
            dense.append(nn.Linear(prev, units))
            dense.append(self._act())
            if CFG.DROPOUT > 0:
                dense.append(nn.Dropout(CFG.DROPOUT))
            prev = units
        dense.append(nn.Linear(prev, n_classes))
        self.head = nn.Sequential(*dense)

    def _act(self):
        act = CFG.ACTIVATION.lower()
        if act == "relu":       return nn.ReLU()
        if act == "leaky_relu": return nn.LeakyReLU()
        if act == "gelu":       return nn.GELU()
        if act == "tanh":       return nn.Tanh()
        if act == "sigmoid":    return nn.Sigmoid()
        return nn.ReLU()

    def forward(self, x):
        # x: (batch, n_qubits)
        x = x.unsqueeze(1)                     # (B, 1, n_qubits)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
# ======================================================================


# ============================= MAIN =============================
def main():
    torch.manual_seed(CFG.RANDOM_STATE)
    np.random.seed(CFG.RANDOM_STATE)

    print("="*70)
    print(" QUANTUM → 1-D CNN CLASSIFIER ".center(70))
    print("="*70)

    df = pd.read_csv(CFG.DATASET_PATH)
    print(f"Loaded → {df.shape}")

    target = find_target_column(df)
    print(f"Target → {target} | Classes = {df[target].nunique()}")

    y = df[target]
    X = df.drop(columns=[target])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=CFG.TEST_SIZE,
        random_state=CFG.RANDOM_STATE, stratify=y_enc
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ---------- Quantum mapping ----------
    if CFG.FORCE_CLASSICAL:
        X_tr_q = X_tr_s
        X_te_q = X_te_s
        print("  → Classical mode (no quantum)")
    else:
        X_tr_q = safe_quantum_mapping(X_tr_s, CFG.N_QUBITS)
        X_te_q = safe_quantum_mapping(X_te_s, CFG.N_QUBITS)

    print(f"Feature map shape → {X_tr_q.shape}")

    # ---------- Torch datasets ----------
    train_ds = TensorDataset(torch.FloatTensor(X_tr_q), torch.LongTensor(y_tr))
    test_ds  = TensorDataset(torch.FloatTensor(X_te_q), torch.LongTensor(y_te))
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False)

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantumCNN(n_qubits=CFG.N_QUBITS, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.LR)

    print(model)
    print(f"Training on {device} | Epochs={CFG.EPOCHS}")

    # ---------- Training loop ----------
    model.train()
    for epoch in range(1, CFG.EPOCHS + 1):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"  Epoch {epoch:02d} | Loss: {epoch_loss/len(train_ds):.6f}")

    # ---------- Evaluation ----------
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average='weighted')

    metrics = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "task_type": "classification",
        "n_classes": int(n_classes)
    }

    print("\n" + "="*70)
    print(f"SUCCESS! ACCURACY: {acc:.6f} | F1: {f1:.6f}")
    print(f"FINAL_METRICS: {json.dumps(metrics)}")
    print(f"MODEL: QuantumFeatureMap({CFG.N_QUBITS}qb) → 1-D CNN")
    print("="*70)

    # === PLOT ONLY OVERALL PERFORMANCE ===
    plot_overall_performance(acc, f1, CFG.PLOT_SAVE_PATH)


# ============================= PLOT FUNCTION =============================
def plot_overall_performance(accuracy, f1_weighted, save_path):
    """Minimal bar chart: Accuracy + Weighted F1"""
    # Ensure directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    metrics = ['Accuracy', 'Weighted F1']
    values = [accuracy, f1_weighted]
    colors = ['#2E8B57', '#4682B4']  # Sea green & Steel blue

    bars = plt.bar(metrics, values, color=colors, width=0.6, edgecolor='black', linewidth=1.2)
    plt.ylim(0, 1.05)
    plt.ylabel('Score', fontsize=12)
    plt.title('Overall Model Performance', fontsize=14, fontweight='bold', pad=20)

    # Add value labels
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overall performance plot saved: {save_path}")
    plt.show()
# ======================================================================


if __name__ == "__main__":
    main()