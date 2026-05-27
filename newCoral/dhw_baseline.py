import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("global_bleaching_environmental.csv", low_memory=False)

# Columns needed
features = ["TSA_DHW"]
target = "Percent_Bleaching"

# Convert to numeric
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop missing values
df = df[features + [target]].dropna()

# Keep valid bleaching range
df = df[(df[target] >= 0) & (df[target] <= 100)]

print("Cleaned Shape:", df.shape)

# ==============================
# TRAIN-TEST SPLIT
# ==============================

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TRAIN LINEAR REGRESSION
# ==============================

model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# EVALUATE
# ==============================

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n===== DHW-ONLY BASELINE RESULTS =====")
print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 4))
