import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS


# 1. Load data
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

os.makedirs("outputs", exist_ok=True)


# 2. Stratified train, validation and test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5,
    random_state=42, stratify=y_temp
)

print("SPLIT CHECKS")
for name, target in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    print(name, "rows:", len(target), "delayed:", int(target.sum()))


# 3. Metrics helper
def get_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        )
    }


# 4. Blind baseline and classifiers
blind_predictions = np.zeros(len(y_val), dtype=int)

model_results = [{
    "model": "blind_all_on_time",
    **get_metrics(y_val, blind_predictions)
}]

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_val)

model_results.append({
    "model": "logistic_regression",
    **get_metrics(y_val, logistic_predictions)
})

# Decision tree depth experiment
tree_results = []
tree_models = {}

for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_val)

    tree_results.append({
        "depth": depth,
        **get_metrics(y_val, predictions)
    })
    tree_models[depth] = tree

tree_depth_df = pd.DataFrame(tree_results)

best_depth = int(
    tree_depth_df.sort_values(
        ["f1", "depth"],
        ascending=[False, True]
    ).iloc[0]["depth"]
)

best_tree = tree_models[best_depth]
model_results.append({
    "model": f"decision_tree_depth_{best_depth}",
    **get_metrics(y_val, best_tree.predict(X_val))
})

# Default Random Forest
rf_default = RandomForestClassifier(
    n_estimators=100, random_state=42
)
rf_default.fit(X_train, y_train)

model_results.append({
    "model": "random_forest_default",
    **get_metrics(y_val, rf_default.predict(X_val))
})

models_df = pd.DataFrame(model_results)
models_df.to_csv("outputs/part_c_models.csv", index=False)


# 5. Feature scores using all 17 features
rf_features = RandomForestClassifier(
    n_estimators=100, random_state=42
)
rf_features.fit(X_train, y_train)

selector_all = SelectKBest(f_classif, k="all")
selector_all.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector_all.scores_,
    "rf_importance": rf_features.feature_importances_
})

feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

feature_scores["is_noise"] = (
    feature_scores["feature"].isin(NOISE_COLUMNS)
)

feature_scores = feature_scores.sort_values(
    "f_rank"
).reset_index(drop=True)

feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv", index=False
)

noise_importance = feature_scores.loc[
    feature_scores["is_noise"], "rf_importance"
].sum()


# 6. SelectKBest experiment
selectk_results = []

for k in [5, 7, 10, 12, 17]:
    model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    selectk_results.append({
        "k": k,
        "val_f1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

selectk_df = pd.DataFrame(selectk_results)
selectk_df.to_csv("outputs/part_c_selectk.csv", index=False)


# 7. Fit selector on training rows only, then transform
selector_7 = SelectKBest(f_classif, k=7)
X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}


# 8. Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)
grid_search.fit(X_train_7, y_train)

grid_predictions = grid_search.best_estimator_.predict(X_val_7)
grid_f1 = f1_score(y_val, grid_predictions, zero_division=0)


# 9. Randomized Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)
random_search.fit(X_train_7, y_train)

random_predictions = random_search.best_estimator_.predict(X_val_7)
random_f1 = f1_score(y_val, random_predictions, zero_division=0)


# 10. Default forest using the same seven features
default_forest_7 = RandomForestClassifier(
    n_estimators=100, random_state=42
)
default_forest_7.fit(X_train_7, y_train)

default_predictions = default_forest_7.predict(X_val_7)
default_f1 = f1_score(y_val, default_predictions, zero_division=0)


# 11. Save tuning comparison
tuning_results = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_f1
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_f1
    }
]

tuning_df = pd.DataFrame(tuning_results)
tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)


# 12. Choose final configuration by validation F1
tie_order = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2
}

tuning_df["tie_order"] = tuning_df["search"].map(tie_order)

winner_row = tuning_df.sort_values(
    ["val_f1", "tie_order"],
    ascending=[False, True]
).iloc[0]

chosen_from = winner_row["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_from == "randomized_search":
    chosen_params = random_search.best_params_
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    }

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(
        random_state=42, **chosen_params
    ))
])

final_model.fit(X_train, y_train)


# 13. Alert threshold selection using validation probabilities
val_probabilities = final_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]:
    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": threshold,
        **get_metrics(y_val, predictions)
    })

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv(
    "outputs/part_c_thresholds.csv", index=False
)

eligible = threshold_df[threshold_df["recall"] >= 0.80]

if len(eligible) > 0:
    chosen_threshold = eligible["threshold"].max()
else:
    chosen_threshold = 0.1


# 14. Evaluate test set exactly once
test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

final_test_df = pd.DataFrame([{
    "threshold": chosen_threshold,
    **get_metrics(y_test, test_predictions)
}])

final_test_df.to_csv(
    "outputs/part_c_final_test.csv", index=False
)


# 15. Save final configuration
final_config_df = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": (
        "unlimited"
        if chosen_params["max_depth"] is None
        else chosen_params["max_depth"]
    ),
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])

final_config_df.to_csv(
    "outputs/part_c_final_config.csv", index=False
)


# 16. Print all results
print("\nCLASSIFIER COMPARISON")
print(models_df.round(4).to_string(index=False))

print("\nFEATURE SCORES")
print(feature_scores.round(4).to_string(index=False))
print("Combined noise importance:", round(noise_importance, 4))

print("\nSELECTK RESULTS")
print(selectk_df.round(4).to_string(index=False))

print("\nTUNING RESULTS")
print(tuning_df.drop(columns="tie_order").round(4).to_string(index=False))

print("\nTHRESHOLD RESULTS")
print(threshold_df.round(4).to_string(index=False))
print("Chosen threshold:", chosen_threshold)

print("\nFINAL TEST RESULTS")
print(final_test_df.round(4).to_string(index=False))

print("\nFINAL CONFIGURATION")
print(final_config_df.to_string(index=False))

print("\nPart C completed successfully!")
