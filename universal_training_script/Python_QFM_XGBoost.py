# -*- coding: utf-8 -*-
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import pennylane as qml

# ------------------- XGBoost -------------------
from xgboost import XGBClassifier

# ------------------- Plotting -------------------
import matplotlib.pyplot as plt

# ============================= CONFIG =============================
class CFG:
    DATASET_PATH     = "/app/upload/dataset.csv"      # your CSV
    TARGET_COLUMN    = None                     # auto-detect
    TEST_SIZE        = 0.2
    RANDOM_STATE     = 42
    N_QUBITS         = 4                       # any integer >= 1
    JITTER           = 1e-6
    FORCE_CLASSICAL  = False                    # skip quantum → raw scaled data

    # ---------------------- XGBoost HYPERPARAMETERS ----------------------
    XGB_N_ESTIMATORS = 200
    XGB_MAX_DEPTH    = 6
    XGB_LEARNING_RATE= 0.1
    XGB_SUBSAMPLE    = 0.9
    XGB_COLSAMPLE    = 0.9

    path_saving_plot = "/app/output"  # Directory for saving plots
    performance_plot_name = "performance_summary.png"
    PLOT_SAVE_PATH   = os.path.join(path_saving_plot, performance_plot_name)
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


# ============================= PLOT FUNCTION =============================
def plot_overall_performance(accuracy, f1_weighted, save_path):
    """Minimal bar chart: Accuracy + Weighted F1"""
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    metrics = ['Accuracy', 'Weighted F1']
    values = [accuracy, f1_weighted]
    colors = ['#2E8B57', '#4682B4']

    bars = plt.bar(metrics, values, color=colors, width=0.6, edgecolor='black', linewidth=1.2)
    plt.ylim(0, 1.05)
    plt.ylabel('Score', fontsize=12)
    plt.title('Overall Model Performance (XGBoost)', fontsize=14,
              fontweight='bold', pad=20)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved: {save_path}")
    plt.show()
# ======================================================================


# ============================= MAIN =============================
def main():
    np.random.seed(CFG.RANDOM_STATE)

    print("="*70)
    print(" QUANTUM FEATURE MAP → XGBOOST CLASSIFIER + PLOT ".center(70))
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

    # ---------- XGBoost model ----------
    if len(np.unique(y_tr)) < 2:
        print("Only 1 class → dummy model")
        pred = np.zeros_like(y_te)
        acc = 0.0
        f1 = 0.0
    else:
        model = XGBClassifier(
            n_estimators=CFG.XGB_N_ESTIMATORS,
            max_depth=CFG.XGB_MAX_DEPTH,
            learning_rate=CFG.XGB_LEARNING_RATE,
            subsample=CFG.XGB_SUBSAMPLE,
            colsample_bytree=CFG.XGB_COLSAMPLE,
            random_state=CFG.RANDOM_STATE,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0
        )
        print("Training XGBoost ...")
        model.fit(X_tr_q, y_tr)

        pred = model.predict(X_te_q)
        acc = accuracy_score(y_te, pred)
        f1  = f1_score(y_te, pred, average='weighted')

    metrics = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "task_type": "classification",
        "n_classes": int(n_classes)
    }

    print("\n" + "="*70)
    print(f"SUCCESS! ACCURACY: {acc:.6f} | F1: {f1:.6f}")
    print(f"FINAL_METRICS: {json.dumps(metrics)}")
    print(f"MODEL: QuantumFeatureMap({CFG.N_QUBITS}qb) → XGBClassifier")
    print("="*70)

    # === PLOT OVERALL PERFORMANCE ===
    plot_overall_performance(acc, f1, CFG.PLOT_SAVE_PATH)


if __name__ == "__main__":
    main()