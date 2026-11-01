
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.shipment_prep import FEATURES


# --------------------------------------------------
# 1. Load data and prepare X and y
# --------------------------------------------------

df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["delivery_hours"]

os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 2. Training, validation and test split
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# --------------------------------------------------
# 3. Scaling without data leakage
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

train_scaled_max_abs_mean = np.abs(
    X_train_scaled.mean(axis=0)
).max()

test_scaled_max_abs_mean = np.abs(
    X_test_scaled.mean(axis=0)
).max()

checks = pd.DataFrame({
    "metric": [
        "train_rows",
        "val_rows",
        "test_rows",
        "train_scaled_max_abs_mean",
        "test_scaled_max_abs_mean"
    ],
    "value": [
        len(X_train),
        len(X_val),
        len(X_test),
        train_scaled_max_abs_mean,
        test_scaled_max_abs_mean
    ]
})

checks.to_csv("outputs/part_b_checks.csv", index=False)


# --------------------------------------------------
# 4. Mean baseline
# --------------------------------------------------

baseline_prediction = np.full(
    len(y_val), y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_val, baseline_prediction
)

baseline_mse = mean_squared_error(
    y_val, baseline_prediction
)

baseline_rmse = np.sqrt(baseline_mse)

baseline_r2 = r2_score(
    y_val, baseline_prediction
)


# --------------------------------------------------
# 5. Decision tree depth experiment
# --------------------------------------------------

depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_results = []

for depth in depths:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    tree_results.append({
        "max_depth": (
            "unlimited" if depth is None else depth
        ),
        "train_rmse": np.sqrt(
            mean_squared_error(y_train, train_prediction)
        ),
        "val_rmse": np.sqrt(
            mean_squared_error(y_val, val_prediction)
        )
    })

tree_depth_df = pd.DataFrame(tree_results)

tree_depth_df.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False
)

# Lowest validation RMSE; ties go to smaller depth.
best_tree_row = tree_depth_df.sort_values(
    ["val_rmse", "max_depth"],
    key=lambda col: col.map(
        lambda value: (
            float("inf") if value == "unlimited"
            else float(value)
        )
    ) if col.name == "max_depth" else col
).iloc[0]

best_depth = (
    None if best_tree_row["max_depth"] == "unlimited"
    else int(best_tree_row["max_depth"])
)


# --------------------------------------------------
# 6. Four candidate models
# --------------------------------------------------

models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "decision_tree": DecisionTreeRegressor(
        max_depth=best_depth,
        random_state=42
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

model_results = [{
    "model": "mean_baseline",
    "split": "val",
    "MAE": baseline_mae,
    "MSE": baseline_mse,
    "RMSE": baseline_rmse,
    "R2": baseline_r2
}]

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_val)

    mse = mean_squared_error(y_val, prediction)

    model_results.append({
        "model": name,
        "split": "val",
        "MAE": mean_absolute_error(y_val, prediction),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_val, prediction)
    })

metrics_df = pd.DataFrame(model_results)


# --------------------------------------------------
# 7. Gradient Boosting learning-rate experiment
# --------------------------------------------------

gb_results = []

for lr in [0.01, 0.1, 1.0]:

    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    gb.fit(X_train, y_train)

    prediction = gb.predict(X_val)

    gb_results.append({
        "learning_rate": lr,
        "val_rmse": np.sqrt(
            mean_squared_error(y_val, prediction)
        ),
        "val_r2": r2_score(y_val, prediction)
    })

gb_learning_df = pd.DataFrame(gb_results)

gb_learning_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False
)


# --------------------------------------------------
# 8. Cross-validation for two best validation models
# --------------------------------------------------

# Rank the four candidates by validation RMSE.
candidate_metrics = metrics_df[
    metrics_df["model"] != "mean_baseline"
]

top_two = candidate_metrics.nsmallest(
    2, "RMSE"
)["model"].tolist()

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# One shared KFold object for both models.
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name in top_two:

    scores = cross_val_score(
        models[name],
        X_train_val,
        y_train_val,
        cv=kf
    )

    cv_results.append({
        "model": name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False
)


# --------------------------------------------------
# 9. Final model and one test evaluation
# --------------------------------------------------

winner = candidate_metrics.sort_values(
    "RMSE"
).iloc[0]["model"]

final_model = models[winner]

# The selected model is already fitted on training rows.
test_prediction = final_model.predict(X_test)

test_mse = mean_squared_error(
    y_test, test_prediction
)

test_row = {
    "model": winner,
    "split": "test",
    "MAE": mean_absolute_error(y_test, test_prediction),
    "MSE": test_mse,
    "RMSE": np.sqrt(test_mse),
    "R2": r2_score(y_test, test_prediction)
}

metrics_df = pd.concat(
    [metrics_df, pd.DataFrame([test_row])],
    ignore_index=True
)

metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)


# --------------------------------------------------
# 10. Print results
# --------------------------------------------------

print("\nSPLIT AND SCALING CHECKS")
print(checks)

print("\nDECISION TREE DEPTH RESULTS")
print(tree_depth_df.round(3))

print("\nMODEL VALIDATION AND FINAL TEST METRICS")
print(metrics_df.round(3))

print("\nGRADIENT BOOSTING LEARNING RATES")
print(gb_learning_df.round(3))

print("\nCROSS-VALIDATION RESULTS")
print(cv_df.round(3))

print("\nSelected final model:", winner)
print("Final test RMSE:", round(test_row["RMSE"], 3))
print("\nPart B completed successfully!")
