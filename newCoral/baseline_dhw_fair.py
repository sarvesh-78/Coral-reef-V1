import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ==============================
# LOAD CLEANED DATA
# ==============================

df = pd.read_csv("cleaned_temporal_dataset.csv")

features = ["TSA_DHW"]
target = "Percent_Bleaching"

X = df[features]
y = df[target]

# SAME SPLIT AS TEMPORAL MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TRAIN BASELINE
# ==============================

model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n===== FAIR DHW BASELINE RESULTS =====")
print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 4))
