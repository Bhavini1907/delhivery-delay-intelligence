
from pathlib import Path
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load original data
    raw = pd.read_csv(DATA_DIR / "raw_shipments.csv")
    hubs = pd.read_csv(DATA_DIR / "hubs.csv")

    print("Raw shipments:", raw.shape)
    print("Hubs:", hubs.shape)

    # Clean shipments
    clean, medians = clean_shipments(raw)

    print("\nClean shipments:", clean.shape)
    print("Medians:", medians)

    # Build features
    modelling = build_features(clean, hubs)

    # Create delay label
    modelling["is_delayed"] = (
        modelling["delivery_hours"] > modelling["promised_hours"]
    ).astype(int)

    # Data quality checks
    checks = pd.DataFrame({
        "check": [
            "raw_rows",
            "clean_rows",
            "duplicates_removed",
            "missing_values",
            "unique_shipment_ids",
            "delayed_shipments",
            "overall_delay_rate"
        ],
        "value": [
            len(raw),
            len(modelling),
            len(raw) - len(modelling),
            int(modelling.isna().sum().sum()),
            int(modelling["shipment_id"].nunique()),
            int(modelling["is_delayed"].sum()),
            float(modelling["is_delayed"].mean())
        ]
    })

    # Delay rate by service type, hub tier and region
    group_results = []

    for column in ["service_type", "hub_tier", "region"]:

        grouped = (
            modelling.groupby(column)
            .agg(
                shipment_count=("shipment_id", "count"),
                delayed_shipments=("is_delayed", "sum"),
                delay_rate=("is_delayed", "mean")
            )
            .reset_index()
            .rename(columns={column: "group"})
        )

        grouped.insert(0, "group_type", column)
        group_results.append(grouped)

    delay_by_group = pd.concat(group_results, ignore_index=True)

    # Numeric correlation matrix
    correlation = (
        modelling.select_dtypes(include="number")
        .corr()
    )

    # Save files
    modelling.to_csv(DATA_DIR / "modelling_table.csv", index=False)
    checks.to_csv(OUTPUT_DIR / "part_a_checks.csv", index=False)
    delay_by_group.to_csv(OUTPUT_DIR / "delay_by_group.csv", index=False)
    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    print("\nSaved modelling table and Part A EDA outputs.")
    print("Total shipments:", len(modelling))
    print("Delayed shipments:", int(modelling["is_delayed"].sum()))
    print("Delay rate:", round(modelling["is_delayed"].mean() * 100, 2), "%")
    print("\nPart A completed successfully.")


if __name__ == "__main__":
    main()
