"""
train_model.py
==============
E-Commerce Analytics and Management System
Author : Tahniyat

Standalone training script — loads data.csv, engineers features,
trains two Random Forest models, evaluates them, and saves to models/.

Usage:
    python train_model.py

Outputs:
    models/rf_revenue_regressor.joblib
    models/rf_churn_classifier.joblib
    models/model_metadata.json
"""

import os
import json
import warnings
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import joblib

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data.csv"
MODELS_DIR  = "models"
PLOTS_DIR   = os.path.join(MODELS_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  E-COMMERCE MODEL TRAINING PIPELINE")
print("=" * 55)
print(f"\n[1/6] Loading dataset from '{DATA_PATH}' ...")
df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
print(f"      Raw shape : {df.shape}")

# ─── 2. Clean Data ────────────────────────────────────────────────────────────
print("\n[2/6] Cleaning data ...")

df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"]     = df["Quantity"] * df["UnitPrice"]
df["CustomerID"]  = df["CustomerID"].astype(int)

print(f"      Clean shape : {df.shape}")
print(f"      Date range  : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
print(f"      Customers   : {df['CustomerID'].nunique():,}")

# ─── 3. Feature Engineering ───────────────────────────────────────────────────
print("\n[3/6] Engineering customer-level features ...")

max_date = df["InvoiceDate"].max()

cust = df.groupby("CustomerID").agg(
    Recency        =("InvoiceDate",  lambda x: (max_date - x.max()).days),
    Frequency      =("InvoiceNo",    "nunique"),
    Monetary       =("Revenue",      "sum"),
    AvgOrderValue  =("Revenue",      "mean"),
    TotalItems     =("Quantity",     "sum"),
    UniqueProducts =("StockCode",    "nunique"),
    AvgUnitPrice   =("UnitPrice",    "mean"),
    NumCountries   =("Country",      "nunique"),
    ActiveDays     =("InvoiceDate",  lambda x: (x.max() - x.min()).days + 1),
    PurchaseSpan   =("InvoiceDate",  lambda x: (x.max() - x.min()).days),
).reset_index()

cust["OrdersPerDay"]    = cust["Frequency"]   / (cust["ActiveDays"] + 1)
cust["RevenuePerOrder"] = cust["Monetary"]    / (cust["Frequency"]  + 1)
cust["ItemsPerOrder"]   = cust["TotalItems"]  / (cust["Frequency"]  + 1)

# Churn label: high recency AND low frequency
median_recency  = cust["Recency"].median()
cust["Churn"]   = (
    (cust["Recency"] > median_recency) & (cust["Frequency"] <= 2)
).astype(int)

print(f"      Feature table : {cust.shape}")
print(f"      Churn rate    : {cust['Churn'].mean()*100:.1f}%  "
      f"({cust['Churn'].sum()} at-risk / {len(cust)} total)")

# ─── 4a. Model A — Random Forest Regressor ────────────────────────────────────
print("\n[4/6] Training Random Forest Regressor (lifetime revenue) ...")

REG_FEATURES = [
    "Recency", "Frequency", "AvgOrderValue", "TotalItems",
    "UniqueProducts", "AvgUnitPrice", "NumCountries",
    "ActiveDays", "PurchaseSpan", "OrdersPerDay", "ItemsPerOrder",
]
REG_TARGET = "Monetary"

X_reg     = cust[REG_FEATURES]
y_reg_log = np.log1p(cust[REG_TARGET])        # log-transform to reduce skew

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg_log, test_size=0.20, random_state=42
)

rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
rf_reg.fit(X_train_r, y_train_r)

# Evaluate
y_pred_log = rf_reg.predict(X_test_r)
y_pred_r   = np.expm1(y_pred_log)
y_true_r   = np.expm1(y_test_r)

mae  = mean_absolute_error(y_true_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_true_r, y_pred_r))
r2   = r2_score(y_true_r, y_pred_r)

cv_r2 = cross_val_score(rf_reg, X_reg, y_reg_log, cv=5, scoring="r2")

print(f"\n      ── Regressor Results ──────────────────────")
print(f"      MAE          : £{mae:>10,.2f}")
print(f"      RMSE         : £{rmse:>10,.2f}")
print(f"      R²           :  {r2:.4f}")
print(f"      5-Fold CV R² :  {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# Plot: Actual vs Predicted + Residuals
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_true_r, y_pred_r, alpha=0.35, color="#2563EB", s=18)
lim = max(float(y_true_r.max()), float(y_pred_r.max()))
axes[0].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")
axes[0].set_xlabel("Actual Revenue (£)")
axes[0].set_ylabel("Predicted Revenue (£)")
axes[0].set_title(f"Actual vs Predicted Revenue  (R²={r2:.4f})")
axes[0].legend()

residuals = y_true_r - y_pred_r
axes[1].hist(residuals, bins=50, color="#7c3aed", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Residual (£)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")

plt.suptitle("Random Forest Regressor — Revenue Prediction", fontsize=13, fontweight="bold")
plt.tight_layout()
reg_eval_path = os.path.join(PLOTS_DIR, "regressor_evaluation.png")
plt.savefig(reg_eval_path, dpi=150)
plt.close()
print(f"      Plot saved → {reg_eval_path}")

# Feature importance
feat_imp_reg = pd.Series(rf_reg.feature_importances_, index=REG_FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_imp_reg)))
feat_imp_reg.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Importance — Revenue Prediction")
ax.set_xlabel("Importance Score")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout()
reg_imp_path = os.path.join(PLOTS_DIR, "regressor_feature_importance.png")
plt.savefig(reg_imp_path, dpi=150)
plt.close()
print(f"      Plot saved → {reg_imp_path}")

# ─── 4b. Model B — Random Forest Classifier ───────────────────────────────────
print("\n[5/6] Training Random Forest Classifier (churn prediction) ...")

CLF_FEATURES = [
    "Recency", "Frequency", "Monetary", "AvgOrderValue",
    "TotalItems", "UniqueProducts", "AvgUnitPrice",
    "ActiveDays", "PurchaseSpan", "OrdersPerDay",
    "RevenuePerOrder", "ItemsPerOrder",
]
CLF_TARGET = "Churn"

X_clf = cust[CLF_FEATURES]
y_clf = cust[CLF_TARGET]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_clf.fit(X_train_c, y_train_c)

# Evaluate
y_pred_c     = rf_clf.predict(X_test_c)
y_pred_proba = rf_clf.predict_proba(X_test_c)[:, 1]
auc          = roc_auc_score(y_test_c, y_pred_proba)
cv_auc       = cross_val_score(rf_clf, X_clf, y_clf, cv=5, scoring="roc_auc")

print(f"\n      ── Classifier Results ─────────────────────")
print(classification_report(y_test_c, y_pred_c,
                             target_names=["Active", "At-Risk"],
                             digits=4))
print(f"      AUC-ROC      :  {auc:.4f}")
print(f"      5-Fold CV AUC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# Plot: Confusion Matrix + ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test_c, y_pred_c,
    display_labels=["Active", "At-Risk"],
    colorbar=False, cmap="Blues", ax=axes[0],
)
axes[0].set_title("Confusion Matrix")

RocCurveDisplay.from_predictions(
    y_test_c, y_pred_proba,
    name=f"RF Classifier  (AUC={auc:.3f})",
    color="#2563EB", ax=axes[1],
)
axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
axes[1].set_title("ROC Curve")
axes[1].legend(loc="lower right")

plt.suptitle("Random Forest Classifier — Churn Prediction", fontsize=13, fontweight="bold")
plt.tight_layout()
clf_eval_path = os.path.join(PLOTS_DIR, "classifier_evaluation.png")
plt.savefig(clf_eval_path, dpi=150)
plt.close()
print(f"      Plot saved → {clf_eval_path}")

# Feature importance
feat_imp_clf = pd.Series(rf_clf.feature_importances_, index=CLF_FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(feat_imp_clf)))
feat_imp_clf.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Importance — Churn Prediction")
ax.set_xlabel("Importance Score")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout()
clf_imp_path = os.path.join(PLOTS_DIR, "classifier_feature_importance.png")
plt.savefig(clf_imp_path, dpi=150)
plt.close()
print(f"      Plot saved → {clf_imp_path}")

# ─── 5. Churn Risk Tiering ────────────────────────────────────────────────────
cust["ChurnProbability"] = rf_clf.predict_proba(X_clf)[:, 1]
cust["ChurnRiskTier"]    = pd.cut(
    cust["ChurnProbability"],
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
)

revenue_at_risk = cust[cust["ChurnRiskTier"] == "High Risk"]["Monetary"].sum()

print("\n      ── Churn Risk Distribution ────────────────")
print(cust["ChurnRiskTier"].value_counts().to_string())
print(f"\n      Revenue at risk (High-Risk customers): £{revenue_at_risk:,.2f}")

# ─── 6. Save Models ───────────────────────────────────────────────────────────
print("\n[6/6] Saving models ...")

reg_path  = os.path.join(MODELS_DIR, "rf_revenue_regressor.joblib")
clf_path  = os.path.join(MODELS_DIR, "rf_churn_classifier.joblib")
meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

joblib.dump(rf_reg, reg_path)
joblib.dump(rf_clf, clf_path)

metadata = {
    "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "dataset":    DATA_PATH,
    "regressor": {
        "file":          reg_path,
        "target":        "Monetary (log-transformed → back-transformed for evaluation)",
        "features":      REG_FEATURES,
        "n_features":    len(REG_FEATURES),
        "n_estimators":  200,
        "max_depth":     10,
        "test_mae":      round(mae,  4),
        "test_rmse":     round(rmse, 4),
        "test_r2":       round(r2,   4),
        "cv_r2_mean":    round(float(cv_r2.mean()),  4),
        "cv_r2_std":     round(float(cv_r2.std()),   4),
    },
    "classifier": {
        "file":              clf_path,
        "target":            "Churn (0=Active, 1=At-Risk)",
        "churn_definition":  "Recency > median AND Frequency <= 2",
        "features":          CLF_FEATURES,
        "n_features":        len(CLF_FEATURES),
        "n_estimators":      200,
        "max_depth":         10,
        "class_weight":      "balanced",
        "test_auc_roc":      round(auc, 4),
        "cv_auc_mean":       round(float(cv_auc.mean()), 4),
        "cv_auc_std":        round(float(cv_auc.std()),  4),
    },
}

with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n      Saved → {reg_path}")
print(f"      Saved → {clf_path}")
print(f"      Saved → {meta_path}")

# ─── Quick load verification ──────────────────────────────────────────────────
loaded_reg = joblib.load(reg_path)
loaded_clf = joblib.load(clf_path)

sample_rev   = np.expm1(loaded_reg.predict(X_test_r.iloc[:3]))
sample_churn = loaded_clf.predict_proba(X_test_c.iloc[:3])[:, 1]

print("\n      ── Load & Inference Verification ──────────")
for i, (r, c) in enumerate(zip(sample_rev, sample_churn)):
    print(f"      Customer {i+1}:  predicted revenue £{r:,.2f}  |  churn risk {c*100:.1f}%")

print("\n" + "=" * 55)
print("  TRAINING COMPLETE ✅")
print("=" * 55)
print(f"\n  Regressor  →  {reg_path}")
print(f"  Classifier →  {clf_path}")
print(f"  Metadata   →  {meta_path}")
print(f"  Plots      →  {PLOTS_DIR}/")
print()
