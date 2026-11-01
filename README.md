
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


## Model results (classification)

### 1. Baseline and model comparison

The blind baseline predicts every shipment as on time. It achieves
76.67% accuracy because most shipments are on time, but it detects
no delayed shipments. Its recall and F1-score are both 0.

The Logistic Regression model achieved validation accuracy of 90.56%,
recall of 83.33%, and F1-score of 0.8046.

The default Random Forest achieved validation accuracy of 90.56%,
recall of 73.81%, and F1-score of 0.7848.

### 2. Feature selection and noise analysis

The five noise features together account for 22.44% of Random Forest
feature importance. All five noise features ranked outside the top
seven according to the F-score ranking.

The seven highest-ranked features were:

- distance_km
- hub_tier
- hub_daily_capacity
- n_stops
- promised_hours
- rain_flag
- sale_event_flag

All seven had F-scores above 30.

### 3. SelectKBest comparison

| Selected features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

Using seven features produced the highest validation F1-score of 0.8333.
Therefore, k=7 was selected for the final model.

### 4. Hyperparameter tuning

Grid Search evaluated 27 parameter combinations using five-fold
cross-validation (135 total fits). Its best cross-validation accuracy
was 0.8952 and its validation F1-score was 0.7750.

Randomized Search evaluated 9 combinations using five-fold
cross-validation (45 total fits). Its best cross-validation accuracy
was 0.8905 and its validation F1-score was 0.7848.

The default Random Forest using the same seven selected features
achieved a higher validation F1-score of 0.8333. Therefore, the
default model was retained instead of either tuned model.

These results are based on a validation set of 180 shipments, so
small differences should be interpreted cautiously.

### 5. Threshold selection

The final model used a probability threshold of 0.5. This was the
highest tested threshold that maintained validation recall of at
least 0.80.

At this threshold, validation precision was 0.7955, recall was
0.8333, and F1-score was 0.8140.

### 6. Final test evaluation

The untouched test set contained 180 shipments, including 41 delayed
shipments.

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 0.7955 | 0.6889 |
| Recall | 0.8333 | 0.7561 |
| F1-score | 0.8140 | 0.7209 |

Test recall was 7.72 percentage points lower than validation recall.
Because the test set contains only 41 delayed shipments, each
additional correctly detected or missed delayed shipment changes
recall by approximately 2.44 percentage points.

The test results provide an honest estimate of performance on
unseen data. The test set was kept separate from model selection
and threshold tuning.

### 7. Part C output files

- part_c_classification.py
- outputs/part_c_models.csv
- outputs/part_c_feature_scores.csv
- outputs/part_c_selectk.csv
- outputs/part_c_tuning.csv
- outputs/part_c_thresholds.csv
- outputs/part_c_final_test.csv
- outputs/part_c_final_config.csv

## How to run

Clone the repository and install the pinned dependencies:

```bash
git clone https://github.com/Bhavini1907/delhivery-delay-intelligence.git
cd delhivery-delay-intelligence
pip install -r requirements.txt
python generate_data.py
```

Run the Part A cleaning and feature-engineering cells in
`Delhivery_Delay_Intelligence.ipynb` to create the modelling table.

Then run the remaining scripts in this order:

```bash
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`package_model.py` regenerates `models/delay_model.pkl`, the saved
threshold, feature order and imputation values.
`predict_delay.py` generates `outputs/new_shipment_actions.csv`.

The model uses scikit-learn 1.6.1, pinned in requirements.txt.
If the installed scikit-learn version differs, rerun
`python package_model.py` before making predictions.

## Data findings

The cleaned modelling dataset contains 1,200 shipments across
24 hubs, with 50 shipments per hub.

| Segment | Hubs | Mean delivery hours | Delay rate |
|---|---:|---:|---:|
| Standard | 7 | 47.24 | 2.57% |
| Watch | 8 | 56.81 | 15.50% |
| Escalate | 9 | 64.65 | 46.00% |

### K-means separation

| K | Across/within ratio | Cluster sizes |
|---:|---:|---|
| 2 | 2.894 | 15-9 |
| 3 | 5.393 | 9-8-7 |
| 4 | 6.305 | 8-7-5-4 |
| 5 | 6.887 | 8-7-4-3-2 |

K=3 was retained because it provides useful operational groups
without creating very small clusters. K=4 and K=5 have higher
separation ratios but create clusters containing only four,
three or two hubs.

### Tier-versus-segment mismatches

| Hub | City | Tier | Actual segment | Expected segment |
|---|---|---:|---|---|
| HUB07 | Kolkata | 1 | Watch | Standard |
| HUB13 | Nagpur | 2 | Escalate | Watch |
| HUB15 | Kochi | 2 | Escalate | Watch |
| HUB22 | Dehradun | 3 | Watch | Escalate |

These mismatches show that hub tier alone does not fully
represent observed operational performance.

## Operations recommendation

The final classification model uses a probability threshold of 0.5.
On the 180-row test set, it achieved:

- Accuracy: 86.67%
- Precision: 68.89%
- Recall: 75.61%
- F1-score: 72.09%

This means approximately 7 of every 10 flagged shipments were
actually delayed, while approximately 8 of every 10 delayed
shipments were detected.

### Hubs requiring escalation

The following Escalate-segment hubs have the highest operational
priority based on their observed delay rates:

| Hub | Delay rate |
|---|---:|
| HUB13 | 46% |
| HUB15 | 36% |
| HUB17 | 46% |
| HUB18 | 38% |
| HUB19 | 62% |
| HUB20 | 48% |
| HUB21 | 40% |
| HUB23 | 44% |
| HUB24 | 54% |

The tier mismatches should also be reviewed. HUB07 and HUB22
perform worse or better than their tier implies, while HUB13
and HUB15 are in the Escalate segment despite being Tier 2.
Hub management should investigate capacity, route distance,
stop counts and local operating conditions.

### New shipment actions

The prediction pipeline processed all 48 new shipments:

- 8 shipments were flagged.
- 6 flagged shipments require escalation to a hub manager.
- 2 flagged shipments require a proactive customer call.
- 40 shipments require no action.

The alert threshold is 0.5. A flagged shipment assigned to an
Escalate hub is sent to the hub manager; other flagged shipments
receive a proactive customer call.

## Production plan

The following practices support reliable production use:

| Area | Project implementation |
|---|---|
| Accuracy | Test-set precision, recall, F1 and accuracy are reported. |
| Reliability | The packaged model is reloaded and checked against the original model. |
| Reproducibility | Random states, stratified splits and dependency versions are fixed. |
| Performance | The pipeline uses a compact set of selected features. |
| Maintainability | Cleaning and feature engineering are shared through `src/shipment_prep.py`. |
| Consistency | `feature_order.csv` preserves the training feature order during prediction. |
| Monitorability | Prediction outputs and operational metrics are saved as CSV files. |

### Monitoring and alert triggers

Monitor these quantities at least weekly:

1. Delayed shipment rate: investigate if it exceeds 20% or rises by 5 percentage points from the established baseline.
2. Model recall: review the model if recall falls below 70%.
3. Missing input values: investigate if more than 5% of values in an input feature are missing.
4. Feature drift: investigate if the Population Stability Index (PSI) exceeds 0.20 for distance_km or promised_hours.
5. Flagged shipment share: review operations if more than 30% of shipments are flagged.

These are proposed initial triggers and should be calibrated using
real production data.

### Retraining and data drift

Retrain the model quarterly, or earlier if monitoring shows sustained
performance deterioration or a major operational change. Evaluate
each retrained model on a time-based holdout before deployment.

The sale period from 15 to 28 September should be compared with the
same sale-event period in other years or comparable sale periods,
rather than ordinary non-sale weeks. Use `sale_event_flag` to
distinguish sale-related demand from normal operations.

Potential drift sources include changes in delivery distance, stop
counts, service-type mix, hub capacity, rainfall and sale volumes.

## Limitations

1. **Synthetic data:** The dataset is synthetic, so its patterns and model performance may not represent real delivery operations.
2. **Small test sample:** The test set contains only 41 delayed shipments. Precision and recall estimates may change with a larger dataset.
3. **Threshold selection:** The 0.5 classification threshold was selected using one validation split. Different samples may produce a different optimal threshold.
4. **Random data split:** A random stratified split does not fully simulate predicting future shipments. A time-based evaluation is recommended before production deployment.
5. **Fixed number of clusters:** K-means was configured to produce three hub segments. Actual hub behaviour may be more varied or may change over time.
6. **Correlation is not causation:** Associations between distance, hub characteristics and delays do not prove that these factors directly cause delays.
7. **Proposed monitoring triggers:** The production thresholds are starting recommendations and require calibration against real operational data.
