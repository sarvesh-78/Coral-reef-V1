import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(
    "global_bleaching_environmental.csv",
    low_memory=False
)

print("Original Shape:", df.shape)

# ==============================
# SELECT FEATURES (Added Latitude)
# ==============================

features = [
    "SSTA_DHW",
    "TSA_DHW",
    "SSTA",
    "SSTA_Frequency",
    "Temperature_Maximum",
    "Turbidity",
    "Depth_m"
]
target = "Percent_Bleaching"

columns_to_use = features + [target]

# ==============================
# FORCE NUMERIC CONVERSION
# ==============================

for col in columns_to_use:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df_model = df[columns_to_use].dropna()

print("After Numeric Conversion + Drop NA:", df_model.shape)

# Filter realistic bleaching values
df_model = df_model[
    (df_model[target] >= 0) &
    (df_model[target] <= 100)
]

print("After Target Filtering:", df_model.shape)

# ==============================
# SPLIT DATA
# ==============================

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TRAIN MODEL
# ==============================

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# EVALUATE
# ==============================

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n===== REGRESSION RESULTS =====")
print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 4))

# ==============================
# FEATURE IMPORTANCE
# ==============================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(importance_df)

plt.figure(figsize=(8,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df
)

plt.title("Environmental Feature Importance (With Latitude)")
plt.tight_layout()
plt.savefig("env_feature_importance_with_latitude.png", dpi=300)
plt.show()

# ==============================
# SAVE MODEL
# ==============================

model.save_model("xgb_env_model.json")
print("Model saved as xgb_env_model.json")