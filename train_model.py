import os, sys, json, sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from feature_engine import extract_features, engineer_features, FEATURE_NAMES

D = os.path.dirname(os.path.abspath(__file__))


def train_from_csv(csv_path="creditcard.csv"):
    """Train model from creditcard.csv (original dataset)."""
    df = pd.read_csv(csv_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    print(f"[+] Loaded {len(df)} rows, fraud rate: {y.mean()*100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Classification Report:")
    print(classification_report(y_test, pred, target_names=["Safe", "Fraud"]))

    joblib.dump(model, os.path.join(D, "fraud_model.pkl"))
    print("[+] Model saved to fraud_model.pkl")


def train_from_db(db_path=None):
    """Train model from accumulated transactions in SQLite DB."""
    if db_path is None:
        db_path = os.path.join(D, "fraud_detection.db")
    if not os.path.exists(db_path):
        print(f"[!] DB not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT raw_features, COALESCE(label, is_fraud) as lbl FROM transactions WHERE raw_features IS NOT NULL"
    ).fetchall()
    conn.close()

    if len(rows) < 50:
        print(f"[!] Need at least 50 rows, have {len(rows)}")
        return

    X_raw, y = [], []
    for r in rows:
        try:
            feats = json.loads(r["raw_features"])
            X_raw.append(extract_features(feats))
            y.append(int(r["lbl"]))
        except:
            continue

    if len(set(y)) < 2:
        print("[!] Need both fraud and non-fraud samples")
        return

    X_eng = np.array([engineer_features(r) for r in X_raw])
    y = np.array(y)

    # Fit and save scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_eng)
    joblib.dump(scaler, os.path.join(D, "scaler.pkl"))
    print(f"[+] Scaler saved ({X_eng.shape[1]} features)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    fraud_count = int(np.sum(y == 1))
    print(f"[+] Training on {len(y)} samples ({fraud_count} fraud, {len(y)-fraud_count} safe)")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, pred, target_names=["Safe", "Fraud"]))

    joblib.dump(model, os.path.join(D, "fraud_model.pkl"))
    print("[+] Model saved to fraud_model.pkl")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "csv"
    if mode == "csv":
        csv_path = sys.argv[2] if len(sys.argv) > 2 else "creditcard.csv"
        train_from_csv(csv_path)
    elif mode == "db":
        db_path = sys.argv[2] if len(sys.argv) > 2 else None
        train_from_db(db_path)
    else:
        print(f"Usage: python train_model.py [csv|db] [path]")
        print(f"  csv  - train from creditcard.csv (default)")
        print(f"  db   - train from fraud_detection.db")