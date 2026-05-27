import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("global_bleaching_environmental.csv", low_memory=False)

base_features = [
    "SSTA_DHW",
    "TSA_DHW",
    "SSTA",
    "SSTA_Frequency",
    "Temperature_Maximum",
    "Turbidity",
    "Depth_m"
]

target = "Percent_Bleaching"

columns = ["Site_ID", "Date"] + base_features + [target]

# Convert to numeric
for col in base_features + [target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[columns].dropna()
df = df[(df[target] >= 0) & (df[target] <= 100)]

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Site_ID", "Date"])

print("After initial cleaning:", df.shape)

# ==============================
# TEMPORAL FEATURES (per-site)
# ==============================

df["TSA_DHW_3mean"] = df.groupby("Site_ID")["TSA_DHW"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

df["TSA_DHW_6mean"] = df.groupby("Site_ID")["TSA_DHW"].transform(
    lambda x: x.rolling(6, min_periods=1).mean()
)

df["SSTA_3mean"] = df.groupby("Site_ID")["SSTA"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

df["TSA_DHW_lag1"] = df.groupby("Site_ID")["TSA_DHW"].shift(1)

df = df.dropna()

print("After temporal feature engineering:", df.shape)

features = base_features + [
    "TSA_DHW_3mean",
    "TSA_DHW_6mean",
    "SSTA_3mean",
    "TSA_DHW_lag1"
]

X = df[features]
y = df[target]
groups = df["Site_ID"]  # ⭐ KEY: group by site

# ==============================
# ⭐ GROUPED CROSS-VALIDATION
# ==============================

gkf = GroupKFold(n_splits=5)

r2_scores = []
mae_scores = []

fold = 1

for train_idx, test_idx in gkf.split(X, y, groups):

    print(f"\n--- Fold {fold} ---")

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print("Train samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])

    # ==============================
    # TRAIN XGBOOST
    # ==============================

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ==============================
    # EVALUATION
    # ==============================

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("R2:", round(r2, 4))
    print("MAE:", round(mae, 4))

    r2_scores.append(r2)
    mae_scores.append(mae)

    fold += 1

# ==============================
# FINAL RESULTS
# ==============================

print("\n===== GROUPED SITE-WISE CV RESULTS =====")
print("Mean R2:", round(np.mean(r2_scores), 4))
print("Std R2 :", round(np.std(r2_scores), 4))
print("Mean MAE:", round(np.mean(mae_scores), 4))
print("Std MAE :", round(np.std(mae_scores), 4))