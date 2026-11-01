
# Delhivery Delay Intelligence — Part A

## 1. Data Cleaning

- Original shipment rows: 1,224
- Duplicate rows removed: 24
- Rows after cleaning: 1,200
- Missing distance values: 36
- Missing parcel weight values: 24
- Invalid parcel weight sentinel rows: 5
- Missing declared values: 12
- Missing payment modes: 18
- Service type spellings standardized from 12 to 3.
- Missing values in features and target after cleaning: 0.

## 2. Feature Engineering

Created the following features:

- promised_hours
- is_cod
- is_weekend
- dispatch_hour
- log_declared_value

Hub information was merged using hub_id.

## 3. Delay Analysis

- Total shipments: 1,200
- Delayed shipments: 278
- Overall delay rate: 23.17%

### Finding 1: Service Type

Express shipments have the highest delay rate at 48.97%.
Standard shipments have a delay rate of 20.56%.
Economy shipments have the lowest delay rate at 8.57%.

### Finding 2: Hub Tier

Tier 3 hubs have the highest delay rate at 44%.
Tier 2 hubs have a delay rate of 21.5%.
Tier 1 hubs have the lowest delay rate at 4%.

### Finding 3: Distance

Shipments travelling 601+ km have the highest delay rate at 85.23%.
The 401-600 km band has a delay rate of 43.64%.
The 0-200 km band has the lowest delay rate at 2.19%.

### Finding 4: Top Correlations with Delivery Hours

The three strongest correlations are:

1. promised_hours: 0.8372
2. distance_km: 0.4013
3. n_stops: 0.3247

These are correlations, not proof of causation.

### Finding 5: Log Transformation

The original declared value mean is INR 1533.05,
while its median is INR 1121.50.

After log transformation:
- Log mean: 7.028
- Log median: 7.022

The log transformation compresses large values and reduces
the influence of extreme values.

## 4. Output Files

- data/modelling_table.csv
- outputs/part_a_checks.csv
- outputs/delay_by_group.csv
- outputs/correlation_matrix.csv
