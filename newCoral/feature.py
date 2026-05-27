import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ==============================
# LOAD TRAINED MODEL
# ==============================

model = XGBRegressor()
model.load_model("xgb_env_model_temporal.json")

# ==============================
# DEFINE FEATURE LIST (MUST MATCH TRAINING)
# ==============================

features = [
    "SSTA_DHW",
    "TSA_DHW",
    "SSTA",
    "SSTA_Frequency",
    "Temperature_Maximum",
    "Turbidity",
    "Depth_m",
    "TSA_DHW_3mean",
    "TSA_DHW_6mean",
    "SSTA_3mean",
    "TSA_DHW_lag1"
]

# ==============================
# GET FEATURE IMPORTANCE (GAIN)
# ==============================

booster = model.get_booster()
importance_dict = booster.get_score(importance_type='gain')

# Convert to DataFrame
importance_df = pd.DataFrame({
    "Feature": importance_dict.keys(),
    "Importance": importance_dict.values()
})

# Some features may not appear → add missing ones as 0
for f in features:
    if f not in importance_df["Feature"].values:
        importance_df = pd.concat([
            importance_df,
            pd.DataFrame([{"Feature": f, "Importance": 0}])
        ], ignore_index=True)

# Sort
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(importance_df)
importance_df["Importance"] = (
    importance_df["Importance"] / importance_df["Importance"].sum()
)

# ==============================
# PLOT
# ==============================

plt.figure(figsize=(8,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Normalized Gain Importance")

plt.tight_layout()
plt.savefig("env_feature_importance.png", dpi=300)
plt.show()
