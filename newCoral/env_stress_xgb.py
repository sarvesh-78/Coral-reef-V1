import pandas as pd
import numpy as np
from scipy.stats import pearsonr
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

for col in base_features + [target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[columns].dropna()
df = df[(df[target] >= 0) & (df[target] <= 100)]

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Site_ID", "Date"])

print("Cleaned Shape:", df.shape)

# ==============================
# ADD TEMPORAL FEATURES (Per Site)
# ==============================

df["TSA_DHW_3mean"] = df.groupby("Site_ID")["TSA_DHW"].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

df["TSA_DHW_6mean"] = df.groupby("Site_ID")["TSA_DHW"].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)

df["SSTA_3mean"] = df.groupby("Site_ID")["SSTA"].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

df["TSA_DHW_lag1"] = df.groupby("Site_ID")["TSA_DHW"].shift(1)

df = df.dropna()

features = base_features + [
    "TSA_DHW_3mean",
    "TSA_DHW_6mean",
    "SSTA_3mean",
    "TSA_DHW_lag1"
]

# ==============================
# LOAD TRAINED MODEL
# ==============================

model = XGBRegressor()
model.load_model("xgb_env_model_temporal.json")

print("Temporal XGBoost model loaded.")

# ==============================
# MULTI-SITE VALIDATION
# ==============================

site_counts = df["Site_ID"].value_counts()
valid_sites = site_counts[site_counts >= 20].index

results = []

for site_id in valid_sites:

    df_site = df[df["Site_ID"] == site_id].copy()

    if len(df_site) < 6:
        continue

    X_site = df_site[features]

    y_pred = model.predict(X_site)
    stress_score = y_pred / 100
    bleach_actual = df_site[target] / 100

    if np.std(stress_score) == 0 or np.std(bleach_actual) == 0:
        continue

    r, p = pearsonr(stress_score, bleach_actual)
    results.append((site_id, r, p))

# ==============================
# RESULTS
# ==============================

results_df = pd.DataFrame(
    results,
    columns=["Site_ID", "Correlation", "P_value"]
)

mean_r = results_df["Correlation"].mean()
median_r = results_df["Correlation"].median()
significant_pct = (results_df["P_value"] < 0.05).mean() * 100
std_r = results_df["Correlation"].std()

print("\n===== TEMPORAL MODEL VALIDATION =====")
print(f"Total Sites Evaluated: {len(results_df)}")
print(f"Mean Correlation: {mean_r:.3f}")
print(f"Median Correlation: {median_r:.3f}")
print(f"Std Dev Correlation: {std_r:.3f}")
print(f"% Significant Sites (p < 0.05): {significant_pct:.1f}%")
