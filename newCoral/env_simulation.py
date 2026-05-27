import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("global_bleaching_environmental.csv", low_memory=False)

columns = [
    "Site_ID",
    "Date",
    "Temperature_Maximum",
    "Turbidity",
    "Depth_m",
    "TSA_DHW",
    "SSTA",
    "SSTA_DHW",
    "SSTA_Frequency",
    "Percent_Bleaching"
]

for col in columns[2:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[columns].dropna()

print("Cleaned Shape:", df.shape)

# ==============================
# FILTER VALID SITES (>=20 records)
# ==============================

site_counts = df["Site_ID"].value_counts()
valid_sites = site_counts[site_counts >= 20].index

print("Total valid sites:", len(valid_sites))

results = []

# ==============================
# LOOP THROUGH SITES
# ==============================

for site_id in valid_sites:

    df_site = df[df["Site_ID"] == site_id].copy()
    df_site = df_site.sort_values("Date")

    if len(df_site) < 6:
        continue

    features = [
        "TSA_DHW",
        "Temperature_Maximum",
        "Turbidity",
        "Depth_m",
        "SSTA",
        "SSTA_DHW",
        "SSTA_Frequency"
    ]

    # --------------------------
    # Percentile Normalization
    # --------------------------

    for col in features:
        low = df_site[col].quantile(0.05)
        high = df_site[col].quantile(0.95)

        df_site[col] = df_site[col].clip(low, high)

        if high > low:
            df_site[col + "_norm"] = (df_site[col] - low) / (high - low)
        else:
            df_site[col + "_norm"] = 0

    # Invert depth (shallower = more stress)
    df_site["Depth_m_norm"] = 1 - df_site["Depth_m_norm"]

    # --------------------------
    # Thermal Accumulation
    # --------------------------

    df_site["TSA_DHW_norm_accum"] = (
        df_site["TSA_DHW_norm"]
        .rolling(window=4, min_periods=1)
        .mean()
    )

    # --------------------------
    # Weighted Stress Score
    # --------------------------

    weights = {
        "TSA_DHW_norm_accum": 0.35,
        "Temperature_Maximum_norm": 0.15,
        "Turbidity_norm": 0.10,
        "Depth_m_norm": 0.08,
        "SSTA_norm": 0.12,
        "SSTA_DHW_norm": 0.10,
        "SSTA_Frequency_norm": 0.10
    }

    df_site["S_env"] = 0

    for feature, weight in weights.items():
        df_site["S_env"] += weight * df_site[feature]

    # --------------------------
    # Correlation with Bleaching
    # --------------------------

    bleach = df_site["Percent_Bleaching"] / 100
    stress = df_site["S_env"]

    try:
        r, p = pearsonr(stress, bleach)
        results.append((site_id, r, p))
    except:
        continue

# ==============================
# RESULTS ANALYSIS
# ==============================

results_df = pd.DataFrame(
    results,
    columns=["Site_ID", "Correlation", "P_value"]
)

mean_r = results_df["Correlation"].mean()
median_r = results_df["Correlation"].median()
significant_pct = (results_df["P_value"] < 0.05).mean() * 100

print("\n===== Multi-Site Validation =====")
print(f"Total Sites Evaluated: {len(results_df)}")
print(f"Mean Correlation: {mean_r:.3f}")
print(f"Median Correlation: {median_r:.3f}")
print(f"% Significant Sites (p < 0.05): {significant_pct:.1f}%")
