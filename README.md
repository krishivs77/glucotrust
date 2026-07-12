# GlucoTrust

Reliability-aware blood glucose forecasting from continuous glucose monitoring, meal, and insulin time-series data.

## Overview

GlucoTrust is a clinical machine learning project for short-term blood glucose forecasting. The project predicts future glucose values from recent continuous glucose monitoring (CGM), meal, insulin, and basal-rate data, while also estimating when forecasts may be less reliable.

The goal is not only to output a glucose prediction, but to study whether model uncertainty can help identify forecasts that should be trusted less. This is especially important in safety-relevant settings such as hypoglycemia and hyperglycemia risk.

## Research Question

Can short-term blood glucose be forecast from recent CGM, meal, and insulin data, and can model uncertainty identify which forecasts are less reliable?

## Project Goals

- Build a reproducible glucose forecasting pipeline from raw diabetes time-series XML data.
- Predict blood glucose 30 and 60 minutes into the future.
- Compare persistence, Ridge regression, Random Forest, XGBoost, and bootstrapped XGBoost ensemble models.
- Evaluate whether meal and insulin context improves forecasting beyond CGM-only history.
- Estimate forecast uncertainty using ensemble disagreement.
- Test whether uncertainty can support selective prediction and abstention.
- Extend the project with patient-specific evaluation, wearable/life-event features, glucose-risk classification, and uncertainty attribution.

## Dataset

This project uses the official OhioT1DM dataset for blood glucose prediction research.

The current pipeline uses both the 2018 and 2020 OhioT1DM cohorts.

Current dataset summary:

- 12 patients
- 24 XML files
- 12 training files
- 12 testing files
- 2018 cohort: 6 patients
- 2020 cohort: 6 patients
- Approximately 39-53 training days per patient
- Approximately 9.5-13.9 testing days per patient
- 1,507,154 parsed events
- 166,533 CGM glucose readings
- 2,168 meal events
- 3,733 bolus insulin events

Parsed event streams include:

- CGM glucose readings
- Finger-stick glucose measurements
- Basal insulin events
- Temporary basal insulin events
- Bolus insulin events
- Meal/carbohydrate events
- Sleep events
- Work events
- Stressor events
- Hypoglycemia events
- Illness events
- Exercise events
- Wearable heart rate
- Wearable GSR
- Wearable skin temperature
- Wearable air temperature
- Wearable steps
- Wearable sleep states
- Acceleration streams

## Dataset Source

This project uses the OhioT1DM dataset for blood glucose level prediction. The dataset was obtained directly from the original OhioT1DM dataset source after requesting access.

Raw data files are not included in this repository. To reproduce the analysis, users should request access to the OhioT1DM dataset from the original dataset source and place the XML files under `data/raw/OhioT1DM/`.

Original dataset reference:

Marling, C., & Bunescu, R. (2020). *The OhioT1DM Dataset for Blood Glucose Level Prediction*.

## Current Data Pipeline

The pipeline converts raw XML event streams into machine-learning-ready forecasting datasets.

Raw XML files are first summarized in a manifest, then parsed into event-level tables. CGM readings are resampled into a regular 5-minute timeline, lagged glucose features are generated, and meal/insulin event summaries are aligned to each timestamp without using future information.

Pipeline:

1. Raw XML files
2. XML file manifest
3. Parsed event tables
4. 5-minute CGM timeline
5. CGM-only lagged forecasting dataset
6. CGM + meal/insulin context forecasting dataset
7. Baseline model comparison
8. XGBoost ensemble uncertainty analysis
9. Selective prediction analysis
10. Visualization/report generation

Implemented scripts:

- `src/data/inspect_xml_files.py`
- `src/data/build_manifest.py`
- `src/data/parse_xml_events.py`
- `src/data/build_cgm_timeline.py`
- `src/features/build_cgm_lag_dataset.py`
- `src/features/build_context_dataset.py`
- `src/models/train_cgm_baselines.py`
- `src/models/train_context_baselines.py`
- `src/models/train_xgb_ensemble_uncertainty.py`
- `src/models/train_context_xgb_ensemble_uncertainty.py`
- `src/evaluation/selective_prediction.py`
- `src/evaluation/context_selective_prediction.py`
- `src/visualization/plot_cgm_baseline_results.py`
- `src/visualization/plot_uncertainty_bins.py`
- `src/visualization/plot_selective_prediction.py`
- `src/visualization/plot_context_comparison.py`
- `src/visualization/plot_context_selective_prediction.py`

## Prediction Setup

The project evaluates short-term glucose forecasting as a time-series regression task.

For each timestamp, the model uses recent patient history and available context to predict future glucose.

- Sampling interval: 5 minutes
- CGM input window: previous 2 hours
- Lag features: 0 to 120 minutes
- Forecast horizons: 30 minutes and 60 minutes
- Main task: future glucose regression

At 5-minute sampling:

- Past 2 hours = 24 previous time steps
- 30-minute forecast = 6 steps ahead
- 60-minute forecast = 12 steps ahead

All feature engineering is performed using information available at or before the prediction timestamp. Future glucose values are used only as supervised learning targets.

## Processed Datasets

### CGM-only dataset

The CGM-only supervised dataset contains:

- 168,378 usable forecasting windows
- 136,565 training rows
- 31,813 testing rows
- 32 CGM-only lag/trend features
- 2 regression targets:
  - `target_glucose_30min`
  - `target_glucose_60min`

CGM-only features include:

- glucose lag values from 0 to 120 minutes
- glucose change over 30, 60, and 120 minutes
- rolling glucose mean over 30 and 60 minutes
- rolling glucose standard deviation over 30 and 60 minutes

### CGM + meal/insulin context dataset

The context dataset keeps the same forecasting windows and adds 26 meal/insulin context features.

Context features include:

- carbohydrates in the last 30, 60, 120, and 180 minutes
- meal counts in the last 30, 60, 120, and 180 minutes
- bolus insulin units in the last 30, 60, 120, and 180 minutes
- bolus event counts in the last 30, 60, 120, and 180 minutes
- bolus calculator carbohydrate input in the last 30, 60, 120, and 180 minutes
- time since last meal
- time since last bolus
- current basal insulin rate
- indicators for whether prior meal, bolus, and basal information are known

The combined dataset contains:

- 168,378 usable forecasting windows
- 136,565 training rows
- 31,813 testing rows
- 58 total model features
- 32 CGM-only features
- 26 meal/insulin context features

## CGM-Only Baseline Results

The first baseline uses only past CGM glucose values, without meal, insulin, exercise, or wearable features.

Models evaluated:

- Persistence baseline
- Ridge regression
- Random Forest regression
- XGBoost regression

The persistence baseline predicts that future glucose will equal current glucose.

| Model | 30-min MAE | 30-min RMSE | 30-min R² | 60-min MAE | 60-min RMSE | 60-min R² |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 17.20 | 24.23 | 0.839 | 28.93 | 39.46 | 0.573 |
| Ridge Regression | 15.00 | 21.24 | 0.876 | 26.37 | 35.15 | 0.661 |
| Random Forest | 14.39 | 20.73 | 0.882 | 25.34 | 34.49 | 0.673 |
| XGBoost | 14.54 | 20.84 | 0.881 | 25.36 | 34.33 | 0.676 |

![CGM-only baseline forecasting error](reports/figures/cgm_baseline_mae.png)

The machine learning models improve over the persistence baseline for both forecast horizons. The improvement is larger for 60-minute prediction, where current glucose alone becomes a weaker baseline.

In the CGM-only setting, Random Forest performed best at the 30-minute horizon, while XGBoost performed best at the 60-minute horizon.

## CGM + Meal/Insulin Context Results

Meal and insulin context features were added to test whether recent food and insulin events improve forecasting beyond glucose momentum alone.

| Model | Feature Set | 30-min MAE | 30-min RMSE | 30-min R² | 60-min MAE | 60-min RMSE | 60-min R² |
|---|---|---:|---:|---:|---:|---:|---:|
| Persistence | CGM only | 17.20 | 24.23 | 0.839 | 28.93 | 39.46 | 0.573 |
| Ridge Regression | CGM only | 15.00 | 21.24 | 0.876 | 26.37 | 35.15 | 0.661 |
| Random Forest | CGM only | 14.39 | 20.73 | 0.882 | 25.34 | 34.49 | 0.673 |
| XGBoost | CGM only | 14.54 | 20.84 | 0.881 | 25.36 | 34.33 | 0.676 |
| Ridge Regression | CGM + context | 14.54 | 20.67 | 0.883 | 25.46 | 34.07 | 0.681 |
| Random Forest | CGM + context | 14.06 | 20.25 | 0.887 | 24.68 | 33.58 | 0.690 |
| XGBoost | CGM + context | 14.24 | 20.54 | 0.884 | 24.53 | 33.46 | 0.693 |

![Context vs CGM-only MAE, 30-minute forecast](reports/figures/context_vs_cgm_mae_30min.png)

![Context vs CGM-only MAE, 60-minute forecast](reports/figures/context_vs_cgm_mae_60min.png)

Adding meal and insulin context improved average forecasting error across model families. The improvement was more pronounced for the 60-minute horizon, where meal and insulin effects have more time to influence future glucose.

For XGBoost, adding context improved 60-minute MAE from 25.36 to 24.53 mg/dL. For Random Forest, adding context improved 60-minute MAE from 25.34 to 24.68 mg/dL.

Although the global MAE gains are modest, the context features make the model more clinically meaningful by allowing it to use recent carbohydrate intake, insulin delivery, and basal-rate information instead of relying only on glucose momentum.

## Ensemble Uncertainty

To estimate forecast uncertainty, the project trains ensembles of 10 XGBoost models using bootstrapped training samples and different random seeds.

For each test prediction:

- `ensemble_mean` = average prediction across models
- `ensemble_std` = standard deviation across models

The ensemble mean is used as the final forecast. The ensemble standard deviation is used as an uncertainty estimate.

## CGM-Only Ensemble Reliability

| Target | MAE | RMSE | R² | Uncertainty-error Spearman |
|---|---:|---:|---:|---:|
| 30-min glucose | 14.56 | 20.88 | 0.880 | 0.234 |
| 60-min glucose | 25.38 | 34.33 | 0.676 | 0.184 |

The CGM-only ensemble achieved performance similar to the single XGBoost model. Ensemble uncertainty was positively associated with actual forecast error, especially for the 30-minute forecast.

## Context Ensemble Reliability

| Target | MAE | RMSE | R² | Uncertainty-error Spearman |
|---|---:|---:|---:|---:|
| 30-min glucose | 14.21 | 20.51 | 0.885 | 0.296 |
| 60-min glucose | 24.44 | 33.32 | 0.695 | 0.238 |

Adding meal and insulin context improved both forecasting accuracy and uncertainty-error alignment.

Compared with the CGM-only ensemble:

- 30-minute MAE improved from 14.56 to 14.21.
- 60-minute MAE improved from 25.38 to 24.44.
- 30-minute uncertainty-error Spearman improved from 0.234 to 0.296.
- 60-minute uncertainty-error Spearman improved from 0.184 to 0.238.

This suggests that meal and insulin context improves not only point prediction accuracy, but also the usefulness of ensemble disagreement as a reliability signal.

## Uncertainty Bins

Predictions were sorted by ensemble uncertainty and split into five equal-frequency bins:

- very low uncertainty = most confident 20% of predictions
- low uncertainty = next 20%
- medium uncertainty = middle 20%
- high uncertainty = next 20%
- very high uncertainty = least confident 20% of predictions

### CGM-only uncertainty bins

For the CGM-only ensemble, higher uncertainty bins generally had larger forecast errors.

| Target | Uncertainty Bin | RMSE |
|---|---|---:|
| 30-min glucose | Very low | 14.88 |
| 30-min glucose | Low | 16.66 |
| 30-min glucose | Medium | 19.28 |
| 30-min glucose | High | 21.87 |
| 30-min glucose | Very high | 28.81 |
| 60-min glucose | Very low | 27.47 |
| 60-min glucose | Low | 29.67 |
| 60-min glucose | Medium | 32.13 |
| 60-min glucose | High | 36.04 |
| 60-min glucose | Very high | 43.90 |

![Forecast error by uncertainty bin](reports/figures/xgb_uncertainty_bins_mae.png)

### Context uncertainty bins

For the context ensemble, uncertainty bins also separated lower-error and higher-error forecasts.

| Target | Uncertainty Bin | RMSE |
|---|---|---:|
| 30-min glucose | Very low | 13.46 |
| 30-min glucose | Low | 15.16 |
| 30-min glucose | Medium | 17.78 |
| 30-min glucose | High | 22.53 |
| 30-min glucose | Very high | 29.46 |
| 60-min glucose | Very low | 24.01 |
| 60-min glucose | Low | 27.18 |
| 60-min glucose | Medium | 30.57 |
| 60-min glucose | High | 36.52 |
| 60-min glucose | Very high | 44.38 |

This separation indicates that ensemble disagreement is useful for distinguishing more reliable forecasts from less reliable forecasts.

## Selective Prediction

Selective prediction evaluates whether the model can improve reliability by abstaining from forecasts with high ensemble uncertainty.

Predictions are ranked from lowest uncertainty to highest uncertainty. Then the model keeps only the most confident predictions at each coverage level.

For example:

- 100% coverage = keep all predictions
- 80% coverage = keep the most confident 80%, reject the most uncertain 20%
- 60% coverage = keep the most confident 60%, reject the most uncertain 40%
- 40% coverage = keep the most confident 40%, reject the most uncertain 60%
- 30% coverage = keep the most confident 30%, reject the most uncertain 70%

This is not random subsampling. The retained predictions are specifically the predictions with the lowest ensemble uncertainty.

### CGM-only selective prediction

| Target | Coverage | RMSE |
|---|---:|---:|
| 30-min glucose | 100% | 20.88 |
| 30-min glucose | 80% | 18.36 |
| 30-min glucose | 60% | 17.04 |
| 30-min glucose | 40% | 15.80 |
| 30-min glucose | 30% | 15.26 |
| 60-min glucose | 100% | 34.33 |
| 60-min glucose | 80% | 31.49 |
| 60-min glucose | 60% | 29.82 |
| 60-min glucose | 40% | 28.59 |
| 60-min glucose | 30% | 28.09 |

![CGM-only selective prediction RMSE](reports/figures/selective_prediction_rmse.png)

### Context selective prediction

| Target | Coverage | RMSE |
|---|---:|---:|
| 30-min glucose | 100% | 20.51 |
| 30-min glucose | 80% | 17.57 |
| 30-min glucose | 60% | 15.57 |
| 30-min glucose | 40% | 14.34 |
| 30-min glucose | 30% | 13.82 |
| 60-min glucose | 100% | 33.32 |
| 60-min glucose | 80% | 29.93 |
| 60-min glucose | 60% | 27.38 |
| 60-min glucose | 40% | 25.64 |
| 60-min glucose | 30% | 24.97 |

![Selective prediction comparison, 30-minute forecast](reports/figures/selective_prediction_context_comparison_30min.png)

![Selective prediction comparison, 60-minute forecast](reports/figures/selective_prediction_context_comparison_60min.png)

Context-aware selective prediction produced a cleaner coverage-error tradeoff than the CGM-only version, especially for 60-minute forecasts.

At 60 minutes, the context ensemble reduced RMSE from 33.32 at full coverage to 24.97 when retaining the most confident 30% of predictions. This supports the main reliability claim of the project: ensemble uncertainty can help identify which glucose forecasts are more likely to be trustworthy.

## Current Interpretation

The current results support four findings:

1. Past glucose history provides a strong baseline for short-term glucose forecasting.
2. Machine learning models improve over persistence, especially for 60-minute forecasts.
3. Meal and insulin context modestly improve average forecast accuracy.
4. Context-aware XGBoost ensemble uncertainty improves reliability estimation and supports selective prediction.

The reliability results are the most important part of the project. The model does not simply output a glucose forecast; it also provides an uncertainty signal that can help identify when the forecast is more or less trustworthy.

## Planned Next Steps

### Patient-specific evaluation

Future analysis should evaluate whether model performance and uncertainty behavior differ by patient.

Potential questions:

- Which patients have the highest forecast error?
- Which patients benefit most from context features?
- Does uncertainty-error correlation vary across patients?
- Are high-uncertainty cases concentrated in specific patients or time periods?

### Event-specific evaluation

Future analysis should evaluate whether meal and insulin context helps most during physiologically meaningful time windows.

Potential comparisons:

- no recent meal or bolus
- recent meal only
- recent bolus only
- recent meal and bolus
- recent hypoglycemia event
- recent exercise event

### Add wearable and life-event features

Future versions may add:

- heart rate summaries
- step counts
- sleep indicators
- exercise events
- illness indicators
- stressor indicators
- work indicators
- acceleration features

### Add glucose-risk classification tasks

In addition to regression forecasting, the project can define classification tasks:

- hypoglycemia risk: future glucose below 70 mg/dL
- hyperglycemia risk: future glucose above 180 mg/dL
- glucose zone prediction: low / in-range / high

This would allow evaluation using classification metrics such as AUROC, AUPRC, sensitivity, specificity, F1, and calibration.

### Add uncertainty attribution

Future work will study why predictions are uncertain by testing whether specific features drive prediction instability.

Example output goal:

Prediction uncertainty is high.

Possible uncertainty drivers:

1. Recent glucose trend is unstable.
2. Meal context is missing or inconsistent.
3. Insulin context strongly changes the forecast.
4. Wearable/activity pattern is outside typical training behavior.

## Reproducibility

The raw dataset is not included in this repository. To reproduce the analysis, request access to the OhioT1DM dataset and place the XML files under:

- `data/raw/OhioT1DM/2018/train/`
- `data/raw/OhioT1DM/2018/test/`
- `data/raw/OhioT1DM/2020/train/`
- `data/raw/OhioT1DM/2020/test/`

Then run the scripts in the following order:

1. `src/data/build_manifest.py`
2. `src/data/parse_xml_events.py`
3. `src/data/build_cgm_timeline.py`
4. `src/features/build_cgm_lag_dataset.py`
5. `src/features/build_context_dataset.py`
6. `src/models/train_cgm_baselines.py`
7. `src/models/train_context_baselines.py`
8. `src/models/train_xgb_ensemble_uncertainty.py`
9. `src/models/train_context_xgb_ensemble_uncertainty.py`
10. `src/evaluation/selective_prediction.py`
11. `src/evaluation/context_selective_prediction.py`
12. `src/visualization/plot_cgm_baseline_results.py`
13. `src/visualization/plot_uncertainty_bins.py`
14. `src/visualization/plot_selective_prediction.py`
15. `src/visualization/plot_context_comparison.py`
16. `src/visualization/plot_context_selective_prediction.py`

## Disclaimer

This project is for research and educational purposes only. It is not intended for medical decision-making.
