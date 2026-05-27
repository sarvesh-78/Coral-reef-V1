import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

# =========================
# Load Dataset
# =========================
df = pd.read_csv("global_bleaching_environmental.csv", low_memory=False)

print("Original Shape:", df.shape)

# =========================
# Select Important Columns
# =========================

selected_columns = [
    "Percent_Bleaching",
    "SSTA_DHW",
    "SSTA",
    "Temperature_Maximum",
    "Turbidity",
    "Depth_m",
    "Cyclone_Frequency",
    "Latitude_Degrees"
]

df = df[selected_columns]

# Convert to numeric
for col in selected_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

print("After Cleaning:", df.shape)

# Filter valid bleaching values
df = df[df["Percent_Bleaching"] >= 0]

# =========================
# Simulate Image Probability
# =========================

df["Simulated_Image_Prob"] = df["Percent_Bleaching"] / 100.0

# =========================
# Define Target
# =========================

y = df["Percent_Bleaching"]

# =========================
# Model 1: Environmental Only
# =========================

X_env = df.drop(columns=["Percent_Bleaching", "Simulated_Image_Prob"])

X_train_env, X_test_env, y_train_env, y_test_env = train_test_split(
    X_env, y, test_size=0.2, random_state=42
)

model_env = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model_env.fit(X_train_env, y_train_env)

y_pred_env = model_env.predict(X_test_env)

r2_env = r2_score(y_test_env, y_pred_env)
mae_env = mean_absolute_error(y_test_env, y_pred_env)

print("\n===== Environmental Only Model =====")
print("R2:", round(r2_env, 4))
print("MAE:", round(mae_env, 4))

# =========================
# Model 2: Fusion Model
# =========================

X_fusion = df.drop(columns=["Percent_Bleaching"])

X_train_fus, X_test_fus, y_train_fus, y_test_fus = train_test_split(
    X_fusion, y, test_size=0.2, random_state=42
)

model_fusion = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model_fusion.fit(X_train_fus, y_train_fus)

y_pred_fus = model_fusion.predict(X_test_fus)

r2_fus = r2_score(y_test_fus, y_pred_fus)
mae_fus = mean_absolute_error(y_test_fus, y_pred_fus)

print("\n===== Fusion Model (Env + Image Prob) =====")
print("R2:", round(r2_fus, 4))
print("MAE:", round(mae_fus, 4))

# =========================
# Improvement
# =========================

print("\n===== Improvement Analysis =====")
print("R2 Improvement:", round(r2_fus - r2_env, 4))
print("MAE Reduction:", round(mae_env - mae_fus, 4))

# =========================
# Feature Importance
# =========================

importance = model_fusion.feature_importances_
features = X_fusion.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\n===== Fusion Feature Importance =====")
print(importance_df)
