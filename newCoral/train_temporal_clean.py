import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor



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

print("After initial cleaning:", df.shape)


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

df.to_csv("cleaned_temporal_dataset.csv", index=False)
print("Cleaned dataset saved.")

features = base_features + [
    "TSA_DHW_3mean",
    "TSA_DHW_6mean",
    "SSTA_3mean",
    "TSA_DHW_lag1"
]

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n===== TEMPORAL MODEL RESULTS =====")
print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 4))

model.save_model("xgb_env_model_temporal.json")
print("Temporal model saved.")
