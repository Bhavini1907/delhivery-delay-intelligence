
import numpy as np
import pandas as pd


PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg",
    "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod",
    "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no",
    "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


def clean_shipments(raw, medians=None):

    clean_df = raw.copy()

    # 1. Remove exact duplicates
    clean_df = clean_df.drop_duplicates().copy()

    # 2. Clean service types
    clean_df["service_type"] = (
        clean_df["service_type"].str.strip().str.title()
    )

    # 3. Fill missing payment modes
    clean_df["payment_mode"] = (
        clean_df["payment_mode"].fillna("Unknown")
    )

    # 4. Convert declared value into numeric INR
    clean_df["declared_value_inr"] = (
        clean_df["declared_value"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    clean_df["declared_value_inr"] = pd.to_numeric(
        clean_df["declared_value_inr"]
    )

    clean_df = clean_df.drop(columns=["declared_value"])

    # 5. Convert dispatch time into datetime
    clean_df["dispatch_time"] = pd.to_datetime(
        clean_df["dispatch_time"]
    )

    # 6. Treat scanner faults as missing
    clean_df.loc[
        clean_df["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan

    # 7. Calculate or reuse historical medians
    if medians is None:
        medians = {
            "distance_km": float(clean_df["distance_km"].median()),
            "parcel_weight_kg": float(clean_df["parcel_weight_kg"].median()),
            "declared_value_inr": float(clean_df["declared_value_inr"].median())
        }
    else:
        medians = {
            "distance_km": float(medians["distance_km"]),
            "parcel_weight_kg": float(medians["parcel_weight_kg"]),
            "declared_value_inr": float(medians["declared_value_inr"])
        }

    # 8. Fill missing numerical values
    for column, median_value in medians.items():
        clean_df[column] = clean_df[column].fillna(median_value)

    return clean_df, medians


def build_features(clean, hubs):

    result = clean.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity"
            ]
        ],
        on="hub_id",
        how="left",
        validate="many_to_one"
    )

    # Promised delivery time
    result["promised_hours"] = (
        result["service_type"].map(PROMISED_HOURS).astype(int)
    )

    # COD indicator
    result["is_cod"] = (
        result["payment_mode"] == "COD"
    ).astype(int)

    # Weekend indicator
    result["is_weekend"] = (
        result["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # Dispatch hour
    result["dispatch_hour"] = (
        result["dispatch_time"].dt.hour.astype(int)
    )

    # Log transformation
    result["log_declared_value"] = np.log(
        result["declared_value_inr"]
    )

    return result
