"""
fix_model.py -- Model Compatibility Checker & Re-Saver
Run this ONCE to verify and re-save device_model.pkl for your current environment.

Usage:
    cd backend
    python fix_model.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Suppress version mismatch warnings during load (we handle them below)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
# 1. PRINT ENVIRONMENT INFO
# ------------------------------------------------------------------
print("=" * 60)
print("  ENVIRONMENT DIAGNOSTIC")
print("=" * 60)
print("  Python      : {}".format(sys.version.split()[0]))
print("  NumPy       : {}".format(np.__version__))
import sklearn
print("  Scikit-learn: {}".format(sklearn.__version__))
print("=" * 60)

# ------------------------------------------------------------------
# 2. RESOLVE MODEL PATH
# ------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'device_model.pkl')

print("\n[CHECK] Looking for model at:\n  {}".format(MODEL_PATH))

# ------------------------------------------------------------------
# 3. TRY LOADING EXISTING MODEL
# ------------------------------------------------------------------
loaded_model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            loaded_model = pickle.load(f)

        # Test prediction to confirm model works
        test_input = np.array([[3.3, 45000]])
        test_proba = loaded_model.predict_proba(test_input)[0]

        print("\n[SUCCESS] Model loaded successfully!")
        print("  Model type : {}".format(type(loaded_model).__name__))
        print("  Test input : vcc=3.3, heap=45000")
        print("  Test output: {}".format(test_proba))

        # Re-save with current sklearn/numpy to eliminate version warnings
        print("\n[RESAVE] Re-saving model with current library versions to remove warnings...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(loaded_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("[RESAVE] device_model.pkl re-saved successfully.")
        print("\n[DONE] Model is compatible and ready. Start Flask with:")
        print("       python flask_backend.py\n")

    except Exception as e:
        print("\n[ERROR] Failed to load existing model: {}".format(e))
        print("  --> Proceeding to regenerate a compatible model...\n")
        loaded_model = None
else:
    print("\n[WARNING] device_model.pkl NOT FOUND at expected path.")
    print("  --> Generating a fresh compatible model...\n")

# ------------------------------------------------------------------
# 4. REGENERATE MODEL IF LOAD FAILED OR FILE MISSING
# ------------------------------------------------------------------
if loaded_model is None:
    print("[REGENERATE] Training a new compatible Random Forest model...")

    np.random.seed(42)
    N = 500

    # SAFE: normal voltage + healthy heap
    safe_vcc   = np.random.uniform(3.0, 3.6, N)
    safe_heap  = np.random.uniform(30000, 80000, N)
    safe_label = np.zeros(N, dtype=int)

    # DANGER: low voltage + critically low heap
    danger_vcc   = np.random.uniform(1.8, 2.8, N)
    danger_heap  = np.random.uniform(1000, 15000, N)
    danger_label = np.ones(N, dtype=int)

    # WARNING: borderline values
    warn_vcc   = np.random.uniform(2.8, 3.0, N)
    warn_heap  = np.random.uniform(15000, 30000, N)
    warn_label = np.ones(N, dtype=int)

    X = np.vstack([
        np.column_stack([safe_vcc,   safe_heap]),
        np.column_stack([danger_vcc, danger_heap]),
        np.column_stack([warn_vcc,   warn_heap]),
    ])
    y = np.concatenate([safe_label, danger_label, warn_label])

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[SUCCESS] New compatible model saved to:\n  {}".format(MODEL_PATH))

    # Verify
    with open(MODEL_PATH, 'rb') as f:
        verification = pickle.load(f)

    test_proba = verification.predict_proba(np.array([[3.3, 45000]]))[0]
    print("\n[VERIFY] Test prediction - vcc=3.3, heap=45000")
    print("  Probabilities   : {}".format(test_proba))
    print("  Failure prob    : {}%".format(round(test_proba[1] * 100, 2)))
    print("\n[DONE] Model regenerated successfully.")
    print("       Start Flask with: python flask_backend.py\n")
