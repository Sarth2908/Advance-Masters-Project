import pandas as pd

BIG_FILE = "financial_fraud_detection_dataset.csv"   # Your Kaggle CSV
LABEL_COL = "is_fraud"

print("Reading big file in chunks...")
chunks = pd.read_csv(BIG_FILE, chunksize=200000)

parts = []

for chunk in chunks:
    if chunk[LABEL_COL].dtype == bool:
        chunk[LABEL_COL] = chunk[LABEL_COL].astype(int)
    else:
        chunk[LABEL_COL] = (
            chunk[LABEL_COL]
            .astype(str)
            .str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
        )

    chunk = chunk[chunk[LABEL_COL].isin([0, 1])]
    fraud = chunk[chunk[LABEL_COL] == 1]
    normal = chunk[chunk[LABEL_COL] == 0]

    if len(fraud) == 0 or len(normal) == 0:
        continue

    normal_sample = normal.sample(
        n=min(len(normal), len(fraud) * 5),
        random_state=42,
    )
    parts.append(pd.concat([fraud, normal_sample]))

df = pd.concat(parts)
df = df.sample(n=min(len(df), 100_000), random_state=42)

keep_cols = [
    "transaction_id", "timestamp", "sender_account", "receiver_account",
    "amount", "transaction_type", "merchant_category",
    "location", "device_used", LABEL_COL,
]
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols]

df = df.drop_duplicates()
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df = df.dropna(subset=["amount"])
df[LABEL_COL] = df[LABEL_COL].astype(int)

df.to_csv("fraud_demo_dataset_clean.csv", index=False)
print(f"✓ Saved fraud_demo_dataset_clean.csv with {len(df)} rows")
