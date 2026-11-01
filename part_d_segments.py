
import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 1. Load modelling data and create hub profiles
# --------------------------------------------------

df = pd.read_csv("data/modelling_table.csv")

hub_profile = (
    df.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    )
    .reset_index()
    .sort_values("hub_id")
    .reset_index(drop=True)
)

hub_profile.to_csv("outputs/hub_profile.csv", index=False)


# --------------------------------------------------
# 2. Scale the four clustering features
# --------------------------------------------------

cluster_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    hub_profile[cluster_features]
)


# --------------------------------------------------
# 3. Calculate separation ratios for K=2,3,4,5
# --------------------------------------------------

distance_matrix = pairwise_distances(
    X_scaled,
    metric="euclidean"
)

upper_i, upper_j = np.triu_indices(
    len(hub_profile),
    k=1
)

pair_distances = distance_matrix[upper_i, upper_j]

separation_rows = []

for k in [2, 3, 4, 5]:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(X_scaled)

    same_cluster = (
        labels[upper_i] == labels[upper_j]
    )

    within_mean = pair_distances[same_cluster].mean()
    across_mean = pair_distances[~same_cluster].mean()

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_values(ascending=False)
        .tolist()
    )

    separation_rows.append({
        "K": k,
        "across_over_within": across_mean / within_mean,
        "cluster_sizes": "-".join(
            str(size) for size in cluster_sizes
        )
    })

separation_df = pd.DataFrame(separation_rows)

separation_df["across_over_within"] = (
    separation_df["across_over_within"].round(3)
)

separation_df.to_csv(
    "outputs/part_d_separation.csv",
    index=False
)


# --------------------------------------------------
# 4. Fit final K=3 clustering
# --------------------------------------------------

final_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = final_kmeans.fit_predict(
    X_scaled
)


# --------------------------------------------------
# 5. Name clusters by average delay rate
# --------------------------------------------------

cluster_delay = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

segment_names = {
    cluster_delay.index[0]: "Standard",
    cluster_delay.index[1]: "Watch",
    cluster_delay.index[2]: "Escalate"
}

hub_profile["segment"] = (
    hub_profile["cluster"].map(segment_names)
)


# --------------------------------------------------
# 6. Save hub segments
# --------------------------------------------------

hubs = pd.read_csv("data/hubs.csv")

hub_segments = hub_profile[
    ["hub_id", "segment"]
].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left",
    validate="one_to_one"
)

hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False
)


# --------------------------------------------------
# 7. Save segment profile
# --------------------------------------------------

segment_order = [
    "Standard",
    "Watch",
    "Escalate"
]

segment_profile = (
    hub_profile.groupby("segment")
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
    .reindex(segment_order)
    .reset_index()
)

segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False
)


# --------------------------------------------------
# 8. Find tier-versus-segment mismatches
# --------------------------------------------------

tier_mapping = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(tier_mapping)
)

tier_mismatch = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
].copy()

tier_mismatch = tier_mismatch[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]

tier_mismatch.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False
)


# --------------------------------------------------
# 9. Print summaries
# --------------------------------------------------

print("Part D hub segmentation completed.")

print("\nHub profile shape:", hub_profile.shape)

print("\nSeparation results:")
print(separation_df.to_string(index=False))

print("\nSegment profile:")
print(segment_profile.to_string(index=False))

print("\nTier-versus-segment mismatches:")
print(tier_mismatch.to_string(index=False))

print("\nSaved all Part D segmentation outputs.")
