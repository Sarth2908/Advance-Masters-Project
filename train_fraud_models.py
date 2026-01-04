import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. LOAD DATA
# -------------------------------
CSV_PATH = "fraud_demo_dataset_clean.csv"   # adjust if different

print(f"Loading dataset from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)

print("Raw shape:", df.shape)
print("Columns:", list(df.columns))

# -------------------------------
# 2. BASIC CLEANING
# -------------------------------
# Drop rows without target
df = df.dropna(subset=["is_fraud"])

# Ensure target is int 0/1
df["is_fraud"] = df["is_fraud"].astype(int)

# -------------------------------
# 3. CREATE TIME FEATURES
# -------------------------------
# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# If there are NaT values, drop them
df = df.dropna(subset=["timestamp"])

df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.weekday  # 0=Monday

# -------------------------------
# 4. SELECT FEATURES & TARGET
# -------------------------------
numeric_features = [
    "amount",
    "hour_of_day",
    "day_of_week"
]

categorical_features = [
    "transaction_type",
    "merchant_category",
    "location",
    "device_used"
]

target_col = "is_fraud"

# Filter to only existing columns (safety)
numeric_features = [c for c in numeric_features if c in df.columns]
categorical_features = [c for c in categorical_features if c in df.columns]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# -------------------------------
# 5. ONE-HOT ENCODE CATEGORICALS
# -------------------------------
X_raw = df[numeric_features + categorical_features]
y = df[target_col].values

X_encoded = pd.get_dummies(X_raw, columns=categorical_features, drop_first=True)

print("Encoded shape:", X_encoded.shape)
print("Class distribution (0,1):", np.bincount(y))

# -------------------------------
# 6. TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded.values,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -------------------------------
# 7. HANDLE CLASS IMBALANCE (SMOTE)
# -------------------------------
print("Applying SMOTE ...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Resampled class distribution:", np.bincount(y_train_res))

# -------------------------------
# 8. SCALE NUMERIC FEATURES
# -------------------------------
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 9. TRAIN RANDOM FOREST
# -------------------------------
print("\nTraining RandomForestClassifier ...")
rf_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X_train_res_scaled, y_train_res)

# -------------------------------
# 10. EVALUATE WITH LOWER THRESHOLD
# -------------------------------
print("\nEvaluating RandomForest on TEST set ...")
y_proba = rf_clf.predict_proba(X_test_scaled)[:, 1]

# Default 0.5 threshold
y_pred_05 = (y_proba >= 0.5).astype(int)
print("\n=== Threshold 0.5 ===")
print(confusion_matrix(y_test, y_pred_05))
print(classification_report(y_test, y_pred_05, digits=4))

# More aggressive fraud detection: 0.2 threshold
y_pred_02 = (y_proba >= 0.2).astype(int)
print("\n=== Threshold 0.2 (more frauds caught) ===")
print(confusion_matrix(y_test, y_pred_02))
print(classification_report(y_test, y_pred_02, digits=4))

roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", roc_auc)

# -------------------------------
# 11. SAVE ARTIFACTS
# -------------------------------
print("\nSaving model and preprocessing objects ...")
joblib.dump(scaler, "scaler_fraud.pkl")
joblib.dump(rf_clf, "rf_fraud_model.pkl")
joblib.dump(list(X_encoded.columns), "feature_columns.pkl")

print("Saved: scaler_fraud.pkl, rf_fraud_model.pkl, feature_columns.pkl")
print("Training complete.")
