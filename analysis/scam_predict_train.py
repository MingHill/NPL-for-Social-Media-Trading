#!/usr/bin/env python
"""
TF-IDF  +  function-declaration flags  →  XGBoost-GPU  →  SHAP
==============================================================

CSV schema
----------
code   : Solidity snippet (str)
target : 1 = successful sell, 0 = failed sell
"""
from __future__ import annotations
import pathlib, json, pickle, re, tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# ───────────────────────────────────────────────────────────────
# 0. Configuration
# ───────────────────────────────────────────────────────────────
CSV_PATH  = pathlib.Path("data/code.csv")
MODEL_OUT = pathlib.Path("model_tfidf_flags_xgb.pkl")

TEST_SIZE     = 0.30
VAL_FRACTION  = 0.50
SEED          = 42

FUNCTIONS = [
    "transfer", "transferFrom", "approve", "mint", "burn", "pause",
    "permitAllance", "_burn", "isOwner", "mod", "getHolders", "_math",
    "tokenSymbol", "tokenDecimals", "swap", "executeSwap", "ERC20Coefficient",
    "addToBlacklist", "enableTrading", "removeLimits", "_p76234", "claimGas", "multicall", "execute",
]

TOP_K = 20

# ───────────────────────────────────────────────────────────────
# 1. Load data
# ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
assert {"code", "target"} <= set(df.columns), "CSV must have 'code' and 'target'"

train_df, temp_df = train_test_split(
    df, test_size=TEST_SIZE, stratify=df["target"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=VAL_FRACTION, stratify=temp_df["target"], random_state=SEED
)

def xy(sub):
    return sub["code"].values, sub["target"].values

X_train, y_train = xy(train_df)
X_val,   y_val   = xy(val_df)
X_test,  y_test  = xy(test_df)

# ───────────────────────────────────────────────────────────────
# 2. Function-declaration flags
# ───────────────────────────────────────────────────────────────
FUNCTION_REGEX = re.compile(r"function\s+([A-Za-z0-9_]+)\s*\(", re.IGNORECASE)

def get_function_names(solidity_code: str) -> set[str]:
    return set(FUNCTION_REGEX.findall(solidity_code))

class FunctionPresence(BaseEstimator, TransformerMixin):
    def __init__(self, functions: list[str]):
        self.functions = functions
        self.func_set  = set(functions)

    def fit(self, X, y=None):
        return self

    def transform(self, docs):
        mat = np.zeros((len(docs), len(self.functions)), dtype=np.uint8)
        for i, code in enumerate(docs):
            declared = get_function_names(code)
            for j, fn in enumerate(self.functions):
                if fn in declared:
                    mat[i, j] = 1
        return csr_matrix(mat)

func_flags = FunctionPresence(FUNCTIONS)

# ───────────────────────────────────────────────────────────────
# 3. TF-IDF vectoriser
# ───────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    lowercase=False,
)

# ───────────────────────────────────────────────────────────────
# 4. Pipeline  (TF-IDF ⊕ flags) → XGBoost
# ───────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("features", FeatureUnion([
        ("tfidf", tfidf),
        ("flags", func_flags),
    ])),
    ("xgb", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        device="cuda",
        n_estimators=2500,
        learning_rate=0.05,
        max_depth=12,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=0.1,
        importance_type='gain',  # Explicitly set importance type
        random_state=SEED,
    )),
])

# ───────────────────────────────────────────────────────────────
# 5. Train
# ───────────────────────────────────────────────────────────────
print("\n▶︎ Training …")
pipeline.fit(X_train, y_train)

# ───────────────────────────────────────────────────────────────
# 6. Evaluation
# ───────────────────────────────────────────────────────────────
def report(name, X, y):
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"\n{name} results")
    print("-" * 60)
    print(classification_report(y, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y, y_prob):.4f}")

report("Validation", X_val, y_val)
report("Test", X_test, y_test)

# ───────────────────────────────────────────────────────────────
# 7. Feature Importance Analysis
# ───────────────────────────────────────────────────────────────
print("\n▶︎ Analyzing Feature Importance …")

# Get feature names
tfidf_names = pipeline.named_steps["features"].transformer_list[0][1].get_feature_names_out()
flag_names = [f"has_{fn}" for fn in FUNCTIONS]
feature_names = np.concatenate([tfidf_names, flag_names])

# Get feature importance from XGBoost
feature_importances = pipeline.named_steps["xgb"].feature_importances_

# Ensure feature_importances matches feature_names in length
if len(feature_importances) != len(feature_names):
    print(f"Warning: Feature count mismatch! Importances: {len(feature_importances)}, Names: {len(feature_names)}")
    # Try to fix by truncating the longer one
    min_len = min(len(feature_importances), len(feature_names))
    feature_importances = feature_importances[:min_len]
    feature_names = feature_names[:min_len]

# Sort by importance
sorted_indices = np.argsort(feature_importances)[::-1]

# Print overall feature importance
print(f"\nTop {TOP_K} features by importance:")
print("-" * 60)
for i, idx in enumerate(sorted_indices[:TOP_K], 1):
    print(f"{i:>2}. {feature_names[idx]:<25} {feature_importances[idx]:.6f}")

# Print importance for TF-IDF features
tfidf_indices = np.where(np.arange(len(feature_names)) < len(tfidf_names))[0]
sorted_tfidf = tfidf_indices[np.argsort(feature_importances[tfidf_indices])[::-1]]

print("\nTF-IDF feature importance:")
print("-" * 60)
for i, idx in enumerate(sorted_tfidf[:TOP_K], 1):
    print(f"{i:>2}. {feature_names[idx]:<25} {feature_importances[idx]:.6f}")

# Print importance for function flags
flag_indices = np.where(np.arange(len(feature_names)) >= len(tfidf_names))[0]
sorted_flags = flag_indices[np.argsort(feature_importances[flag_indices])[::-1]]

print("\nFunction flag importance:")
print("-" * 60)
for i, idx in enumerate(sorted_flags, 1):
    print(f"{i:>2}. {feature_names[idx]:<25} {feature_importances[idx]:.6f}")

# ───────────────────────────────────────────────────────────────
# 8. Persist model + metadata
# ───────────────────────────────────────────────────────────────
MODEL_OUT.parent.mkdir(exist_ok=True, parents=True)
with open(MODEL_OUT, "wb") as f:
    pickle.dump(pipeline, f)

meta = {
    "timestamp" : datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "train_size": len(train_df),
    "val_size"  : len(val_df),
    "test_size" : len(test_df),
    "seed"      : SEED,
    "functions" : FUNCTIONS,
    "model_file": MODEL_OUT.name,
    "feature_importances": {
        feature_names[i]: float(feature_importances[i]) 
        for i in sorted_indices[:TOP_K]
    }
}
(MODEL_OUT.with_suffix(".json")).write_text(json.dumps(meta, indent=2))

print(f"\n✅  Saved model to {MODEL_OUT.resolve()}")