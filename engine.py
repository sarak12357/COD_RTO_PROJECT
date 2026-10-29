"""Explainable COD order risk scoring and action selection."""
from pathlib import Path
import re
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

CAT_COLS = ["state", "courier", "utm_source"]
NUM_COLS = ["order_value", "num_items", "customer_prior_orders", "customer_prior_rto",
            "address_len", "name_suspicious", "pincode_missing", "city_missing"]

def add_features(df):
    x = df.copy()
    for col in ["address", "customer_name", "city", "pincode"]:
        if col not in x:
            x[col] = None
    x["address_len"] = x["address"].fillna("").astype(str).str.len()
    x["name_suspicious"] = x["customer_name"].fillna("").astype(str).str.lower().str.contains(
        r"\b(asdf|test|fake|dummy|unknown|abc)\b", regex=True).astype(int)
    x["pincode_missing"] = x["pincode"].isna().astype(int)
    x["city_missing"] = x["city"].isna().astype(int)
    return x

def normalize_outcome(series):
    normalized = series.fillna("").astype(str).str.strip().str.upper()
    return normalized.map({"DELIVERED": 0, "RTO": 1})

def build_model():
    """Create a fresh preprocessing + logistic-regression pipeline."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUM_COLS),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS)
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1500,
                        random_state=42))])

def load_labeled_data(csv_path="data/orders_sample.csv"):
    raw = pd.read_csv(csv_path)
    if "final_status" not in raw:
        raise ValueError("Dataset must contain final_status.")
    y = normalize_outcome(raw["final_status"])
    usable = y.notna()
    df = add_features(raw.loc[usable].copy()).reset_index(drop=True)
    y = y.loc[usable].astype(int).reset_index(drop=True)
    return df, y

def train_model(csv_path="data/orders_sample.csv"):
    """Fit on all labeled data for scoring new orders in the demo UI."""
    df, y = load_labeled_data(csv_path)
    model = build_model()
    model.fit(df[CAT_COLS + NUM_COLS], y)
    return model, df

def historical_oof_predictions(csv_path="data/orders_sample.csv", n_splits=5):
    """Return out-of-fold probabilities; duplicate order IDs stay in the same fold.

    This is a retrospective diagnostic, not a time-based estimate of future performance.
    """
    df, y = load_labeled_data(csv_path)
    if y.nunique() < 2:
        raise ValueError("Historical evaluation requires both DELIVERED and RTO labels.")
    groups = (df["order_id"].fillna(pd.Series(range(len(df)), index=df.index)).astype(str)
              if "order_id" in df else pd.Series(range(len(df))).astype(str))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    x = df[CAT_COLS + NUM_COLS]
    probabilities = pd.Series(index=df.index, dtype=float)
    for train_idx, test_idx in splitter.split(x, y, groups):
        fold_model = build_model()
        fold_model.fit(x.iloc[train_idx], y.iloc[train_idx])
        probabilities.iloc[test_idx] = fold_model.predict_proba(x.iloc[test_idx])[:, 1]
    result = df.copy()
    result["target_rto"] = y
    result["risk_probability"] = probabilities
    result["evaluation_method"] = "5-fold stratified group OOF by order_id; not time-based"
    return result

def choose_action(probability, prior_rto=0, thresholds=None):
    """Thresholds are passed from UI/config; no code changes needed to tune them."""
    t = {"ship_max": 0.20, "confirm_max": 0.36, "cancel_min": 0.50}
    if thresholds:
        t.update(thresholds)
    # Protect first-time buyers: high score alone does not trigger auto-cancellation.
    if probability >= t["cancel_min"] and int(prior_rto or 0) >= 1:
        return "Hold / cancel", "High predicted RTO risk plus previous RTO history; hold for manual review."
    if probability >= t["confirm_max"]:
        return "Nudge to prepaid", "Elevated predicted RTO risk; offer prepaid with a modest discount."
    if probability >= t["ship_max"]:
        return "Ship after confirmation", "Moderate predicted RTO risk; ask the customer to confirm before dispatch."
    return "Ship as-is", "Low predicted RTO risk under the current threshold."

def address_review(row):
    """Deterministic checks only; an optional LLM can add interpretation, not replace these checks."""
    flags = []
    address = str(row.get("address", "") or "").strip()
    name = str(row.get("customer_name", "") or "").strip().lower()
    if len(address) < 15:
        flags.append("Address is very short; verify landmark/street details.")
    if not row.get("city") or pd.isna(row.get("city")):
        flags.append("City is missing.")
    pin = str(row.get("pincode", "") or "").strip()
    if not pin or pin.lower() == "nan" or not re.fullmatch(r"\d{6}(\.0)?", pin):
        flags.append("Pincode is missing or not a six-digit Indian pincode.")
    if re.search(r"\b(asdf|test|fake|dummy|unknown|abc)\b", name):
        flags.append("Name contains a suspicious placeholder keyword; verify with customer.")
    return flags
