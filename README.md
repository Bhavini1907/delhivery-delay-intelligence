
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


## Model results (regression)

### 1. Baseline
- Mean baseline validation R²: -0.021.
- The negative R² means predicting the training-set mean performs slightly worse than predicting the validation-set mean. The baseline does not explain useful variation in delivery times.

### 2. Model comparison
- Linear Regression achieved the lowest validation RMSE of 4.891 hours, with R² of 0.931.
- Gradient Boosting achieved validation RMSE of 4.917 hours, with R² of 0.931.
- Random Forest achieved validation RMSE of 5.510 hours, with R² of 0.913.
- Decision Tree achieved validation RMSE of 6.280 hours, with R² of 0.887.
- Linear Regression was selected because it had the lowest validation RMSE.

### 3. Decision tree depth
- Depth 1 shows high bias: training RMSE is 11.633 and validation RMSE is 12.132. Both errors are high because the tree is too simple.
- Depth 6 gives the best validation RMSE of 6.280 hours.
- Unlimited depth shows high variance: training RMSE falls to 0.000, while validation RMSE rises to 7.837. The tree overfits the training data.
- The increasing gap between training and validation errors at greater depths indicates reduced generalisation.

### 4. Gradient Boosting learning rate
- Learning rate 0.1 performed best, with validation RMSE 4.917 and R² 0.931.
- Learning rate 0.01 learns too slowly with only 100 estimators, resulting in R² 0.713.
- Learning rate 1.0 makes overly aggressive updates and performs worse, with R² 0.844.

### 5. Cross-validation and final model choice
- Linear Regression: mean cross-validation R² 0.926, standard deviation 0.005.
- Gradient Boosting: mean cross-validation R² 0.935, standard deviation 0.004.
- Cross-validation slightly favours Gradient Boosting, while the single validation split favours Linear Regression.
- Linear Regression was selected because the assignment specifies that the final model must have the lowest validation RMSE. Its cross-validation performance is also strong and consistent.

### 6. Final test performance
- Final model: Linear Regression.
- Test MAE: 3.567 hours.
- Test MSE: 20.462.
- Test RMSE: 4.524 hours.
- Test R²: 0.949.
- The model explains approximately 94.9% of the variation in delivery hours on the held-out test set.

### 7. Part B output files
- `outputs/part_b_checks.csv`
- `outputs/part_b_tree_depth.csv`
- `outputs/part_b_metrics.csv`
- `outputs/part_b_gb_learning_rate.csv`
- `outputs/part_b_cv.csv`
- `part_b_regression.py`
