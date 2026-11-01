
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import clean_shipments, FEATURES


# --------------------------------------------------
# 1. Create folders
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 2. Load training data and final configuration
# --------------------------------------------------

df = pd.read_csv("data/modelling_table.csv")
config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

X = df[FEATURES]
y = df["is_delayed"]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

max_depth_value = config["max_depth"]

if str(max_depth_value).lower() == "unlimited":
    max_depth = None
else:
    max_depth = int(max_depth_value)


# --------------------------------------------------
# 3. Rebuild the exact Part C stratified split
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# --------------------------------------------------
# 4. Build and train final pipeline
# --------------------------------------------------

model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
    ("classifier", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    ))
])

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Save trained model
# --------------------------------------------------

joblib.dump(model, "models/delay_model.pkl")


# --------------------------------------------------
# 6. Save threshold
# --------------------------------------------------

pd.DataFrame({
    "threshold": [threshold]
}).to_csv("models/delay_threshold.csv", index=False)


# --------------------------------------------------
# 7. Save exact feature order
# --------------------------------------------------

pd.DataFrame({
    "feature": FEATURES
}).to_csv("models/feature_order.csv", index=False)


# --------------------------------------------------
# 8. Calculate and save training medians
# --------------------------------------------------

raw_shipments = pd.read_csv("data/raw_shipments.csv")

cleaned_shipments, medians = clean_shipments(raw_shipments)

pd.DataFrame({
    "column": list(medians.keys()),
    "median": list(medians.values())
}).to_csv("models/impute_values.csv", index=False)


# --------------------------------------------------
# 9. Reload model and verify probabilities
# --------------------------------------------------

reloaded_model = joblib.load("models/delay_model.pkl")

original_probabilities = model.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

max_difference = np.max(
    np.abs(original_probabilities - reloaded_probabilities)
)


# --------------------------------------------------
# 10. Save reload verification
# --------------------------------------------------

reload_check = pd.DataFrame({
    "metric": [
        "test_rows_scored",
        "reload_max_abs_probability_diff",
        "threshold"
    ],
    "value": [
        len(X_test),
        max_difference,
        threshold
    ]
})

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False
)


# --------------------------------------------------
# 11. Print results
# --------------------------------------------------

print("Model packaging completed.")

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

print("\nSaved files:")
print("models/delay_model.pkl")
print("models/delay_threshold.csv")
print("models/feature_order.csv")
print("models/impute_values.csv")
print("outputs/part_d_reload_check.csv")

print("\nReload verification:")
print(reload_check)
