
import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# --------------------------------------------------
# 1. Load model and saved configuration
# --------------------------------------------------

model = joblib.load("models/delay_model.pkl")

threshold = pd.read_csv(
    "models/delay_threshold.csv"
)["threshold"].iloc[0]

feature_order = pd.read_csv(
    "models/feature_order.csv"
)["feature"].tolist()

impute_df = pd.read_csv("models/impute_values.csv")

medians = dict(zip(
    impute_df["column"],
    impute_df["median"]
))


# --------------------------------------------------
# 2. Load new shipments and hub information
# --------------------------------------------------

new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Preserve original file order
new_shipments["_original_order"] = range(len(new_shipments))


# --------------------------------------------------
# 3. Clean new shipments using training medians
# --------------------------------------------------

cleaned, _ = clean_shipments(
    new_shipments,
    medians=medians
)


# --------------------------------------------------
# 4. Build features using shared function
# --------------------------------------------------

features_df = build_features(cleaned, hubs)


# --------------------------------------------------
# 5. Load features in exact training order
# --------------------------------------------------

X_new = features_df[feature_order]


# --------------------------------------------------
# 6. Predict positive-class probabilities
# --------------------------------------------------

probabilities = model.predict_proba(X_new)[:, 1]

features_df["delay_probability"] = np.round(
    probabilities, 4
)

features_df["flagged"] = (
    features_df["delay_probability"] >= threshold
).astype(int)


# --------------------------------------------------
# 7. Attach hub segments
# --------------------------------------------------

hub_segments = pd.read_csv("outputs/hub_segments.csv")

features_df = features_df.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    validate="many_to_one",
    sort=False
)


# --------------------------------------------------
# 8. Assign operational actions
# --------------------------------------------------

features_df["action"] = np.where(
    features_df["flagged"] == 0,
    "No action",
    np.where(
        features_df["segment"] == "Escalate",
        "Escalate to hub manager",
        "Proactive customer call"
    )
)


# --------------------------------------------------
# 9. Restore original shipment order
# --------------------------------------------------

features_df = features_df.sort_values(
    "_original_order"
)


# --------------------------------------------------
# 10. Save required output columns
# --------------------------------------------------

output_columns = [
    "shipment_id",
    "hub_id",
    "segment",
    "delay_probability",
    "flagged",
    "action"
]

result = features_df[output_columns].copy()

result.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False
)


# --------------------------------------------------
# 11. Print verification
# --------------------------------------------------

print("New shipment predictions completed.")

print("\nOutput shape:", result.shape)
print("\nFlagged shipments:", result["flagged"].sum())

print("\nAction counts:")
print(result["action"].value_counts())

print("\nMissing values:")
print(result.isna().sum())

print("\nFirst five predictions:")
print(result.head())

print("\nSaved: outputs/new_shipment_actions.csv")
