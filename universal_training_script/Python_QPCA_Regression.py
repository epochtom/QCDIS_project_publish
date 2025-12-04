# -*- coding: utf-8 -*-
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

import pennylane as qml

# ------------------- Plotting Imports -------------------
import matplotlib.pyplot as plt

# ============================= CONFIG =============================
class CFG:
    DATASET_PATH   = "/app/upload/dataset.csv"
    TARGET_COLUMN  = None
    TEST_SIZE      = 0.2
    RANDOM_STATE   = 42
    N_QUBITS       = 4
    N_COMPONENTS   = 8
    N_SUBSET       = 4
    MAX_ITER       = 3
    JITTER         = 1e-6
    FORCE_CLASSICAL = False
    path_saving_plot = "/app/output"  # Directory for saving plots
    performance_plot_name = "performance_summary.png"  # Name of performance plot
    PLOT_SAVE_PATH = os.path.join(path_saving_plot, performance_plot_name)  # Full path
# ==================================================================

def find_target_column(df):
    if 'label' in df.columns:
        return 'label'
    if df.iloc[:, 0].nunique() <= 20 and df.iloc[:, 0].dtype in ['int64', 'int32']:
        return df.columns[0]
    return df.columns[-1]

def safe_quantum_pca(X, n_components, n_qubits):
    try:
        print(f"  → Trying Quantum PCA ({n_qubits} qubits, subset={CFG.N_SUBSET})...")
        return quantum_kernel_pca(X, n_components, n_qubits)
    except Exception as e:
        print(f"  → Quantum failed ({e}), falling back to classical PCA")
        pca = PCA(n_components=n_components, random_state=CFG.RANDOM_STATE)
        return pca.fit_transform(X)

def quantum_kernel_pca(X: np.ndarray, n_components: int, n_qubits: int) -> np.ndarray:
    n_samples, n_features = X.shape
    n_components = min(n_components, n_qubits, n_samples)

    if n_features < n_qubits:
        pad = np.zeros((n_samples, n_qubits - n_features))
        X = np.hstack([X, pad])
    else:
        X = X[:, :n_qubits]

    X = np.pi * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
    X += np.random.normal(0, CFG.JITTER, X.shape)

    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    def embedding(x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')

    @qml.qnode(dev, interface="numpy")
    def kernel(x1, x2):
        embedding(x1)
        qml.adjoint(embedding)(x2)
        return qml.expval(qml.PauliZ(0))

    subset_size = min(CFG.N_SUBSET, n_samples)
    idx = np.random.choice(n_samples, subset_size, replace=False)
    X_sub = X[idx]

    K = np.zeros((subset_size, subset_size))
    for i in range(subset_size):
        for j in range(i, subset_size):
            val = kernel(X_sub[i], X_sub[j])
            val = (val + 1.0) / 2.0
            K[i,j] = K[j,i] = val

    one_n = np.ones((subset_size, subset_size)) / subset_size
    K = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    eigvals, eigvecs = np.linalg.eigh(K)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]
    eigvecs /= np.sqrt(eigvals + 1e-12)

    K_full = np.zeros((n_samples, subset_size))
    for i in range(n_samples):
        for j in range(subset_size):
            val = kernel(X[i], X_sub[j])
            K_full[i,j] = (val + 1.0) / 2.0

    row_means = K_full.mean(axis=1, keepdims=True)
    col_means = K.mean(axis=0, keepdims=True)
    global_mean = K.mean()
    K_full = K_full - row_means - col_means + global_mean

    return K_full @ eigvecs

# ============================= MINIMAL PLOT =============================
def plot_overall_performance(accuracy, f1_weighted, save_path):
    """Single clean bar chart: Accuracy + Weighted F1"""
    # Ensure directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    metrics = ['Accuracy', 'Weighted F1']
    values = [accuracy, f1_weighted]
    colors = ['#4CAF50', '#2196F3']

    bars = plt.bar(metrics, values, color=colors, width=0.6, edgecolor='black', linewidth=1.2)
    plt.ylim(0, 1.05)
    plt.ylabel('Score', fontsize=12)
    plt.title('Overall Model Performance', fontsize=14, fontweight='bold', pad=20)

    # Add value labels on top
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overall performance plot saved: {save_path}")
    plt.show()

# ============================= MAIN =============================
def main():
    print("="*70)
    print(" QUANTUM PCA CLASSIFIER + OVERALL PERFORMANCE PLOT ".center(70))
    print("="*70)

    df = pd.read_csv(CFG.DATASET_PATH)
    print(f"Loaded → {df.shape}")

    target = find_target_column(df)
    print(f"Target → {target} | Classes = {df[target].nunique()}")

    y = df[target]
    X = df.drop(columns=[target])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_encoded, test_size=CFG.TEST_SIZE, random_state=CFG.RANDOM_STATE, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    if CFG.FORCE_CLASSICAL:
        pca = PCA(n_components=CFG.N_COMPONENTS, random_state=CFG.RANDOM_STATE)
        X_tr_q = pca.fit_transform(X_tr_s)
        X_te_q = pca.transform(X_te_s)
    else:
        X_tr_q = safe_quantum_pca(X_tr_s, CFG.N_COMPONENTS, CFG.N_QUBITS)
        X_te_q = safe_quantum_pca(X_te_s, CFG.N_COMPONENTS, CFG.N_QUBITS)

    print(f"Reduced → {X_tr_q.shape}")

    if len(np.unique(y_tr)) < 2:
        print("Only 1 class → dummy model")
        pred = np.zeros_like(y_te)
        acc = 0.0
        f1 = 0.0
    else:
        model = LogisticRegression(max_iter=CFG.MAX_ITER, random_state=CFG.RANDOM_STATE)
        model.fit(X_tr_q, y_tr)
        pred = model.predict(X_te_q)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average='weighted')

    print("\n" + "="*70)
    print(f" SUCCESS! Accuracy: {acc:.6f} | F1: {f1:.6f}")
    print(f" MODEL: QuantumKernelPCA({CFG.N_QUBITS}qb) → LogisticRegression")
    print("="*70)

    # === PLOT ONLY OVERALL PERFORMANCE ===
    plot_overall_performance(acc, f1, CFG.PLOT_SAVE_PATH)

if __name__ == "__main__":
    main()