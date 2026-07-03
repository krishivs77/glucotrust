# GlucoTrust

Reliability-aware blood glucose forecasting from continuous glucose monitoring, insulin, meal, and activity data.

## Overview

GlucoTrust is a clinical machine learning project focused on short-term blood glucose forecasting. The goal is not only to predict future glucose values, but also to study when forecasts are unreliable and which input signals may be driving prediction instability.

The project is motivated by the idea that a forecasting model should not simply output a number. It should also communicate when its prediction may be difficult to trust, especially in safety-relevant situations such as hypoglycemia or hyperglycemia risk.

## Research Question

Can short-term blood glucose be forecast from recent CGM, insulin, meal, and activity data, and can uncertainty attribution identify when forecasts should not be trusted?

## Project Goals

- Build a reproducible glucose forecasting pipeline.
- Predict blood glucose 30 and 60 minutes into the future.
- Compare simple baselines, classical machine learning models, and optional neural models.
- Evaluate forecast error, calibration, and uncertainty.
- Analyze patient-specific reliability.
- Study whether feature-level sensitivity can identify inputs that drive forecast instability.

## Prediction Setup

Planned modeling setup:

- Input window: recent patient history, such as the past 2 hours
- Forecast horizons: 30 minutes and 60 minutes
- Main task: future glucose regression
- Secondary task: hypoglycemia / hyperglycemia risk classification

## Planned Models

- Persistence baseline
- Linear regression / ridge regression on lag features
- Random forest
- XGBoost / LightGBM
- Optional LSTM/GRU model

## Current Progress

The project currently supports a CGM-only forecasting baseline. Raw OhioT1DM XML files are parsed into tidy event tables, then CGM readings are resampled into a regular 5-minute timeline. A supervised lag-feature dataset is built using the previous 2 hours of glucose history to predict glucose 30 and 60 minutes into the future.

Current dataset:

- 6 patients
- 12 XML files
- 85,986 usable CGM forecasting windows
- 70,035 training rows
- 15,951 testing rows
- 32 CGM-only lag/trend features

## Initial CGM-Only Results

The first baseline uses only past CGM glucose values, without meal, insulin, exercise, or wearable features.

| Model | 30-min MAE | 30-min RMSE | 30-min R² | 60-min MAE | 60-min RMSE | 60-min R² |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 16.48 | 22.96 | 0.862 | 27.47 | 36.93 | 0.642 |
| Ridge Regression | 14.56 | 20.67 | 0.888 | 25.21 | 33.59 | 0.703 |
| Random Forest | 14.31 | 20.59 | 0.889 | 24.83 | 33.73 | 0.701 |
| XGBoost | 14.28 | 20.63 | 0.888 | 24.58 | 33.44 | 0.706 |

Initial results show that machine learning models improve over the persistence baseline for both forecast horizons. The improvement is modest for 30-minute prediction, where current glucose is already a strong baseline, but becomes more meaningful at 60 minutes. Ridge regression, random forest, and XGBoost perform similarly in the CGM-only setting, suggesting that much of the short-term signal is captured by glucose momentum and recent trend features.

Future work will add meal, insulin, activity, sleep, and wearable features to test whether contextual events improve forecasting, especially during rapid glucose changes and high-error windows.

## Initial Reliability Analysis

To estimate forecast uncertainty, the project trains an ensemble of 10 XGBoost models using bootstrapped training samples and different random seeds. The ensemble mean is used as the final prediction, while the standard deviation across ensemble predictions is used as an uncertainty estimate.

The ensemble achieved similar or slightly better performance than a single XGBoost model:

| Target | MAE | RMSE | R² | Uncertainty-error Spearman |
|---|---:|---:|---:|---:|
| 30-min glucose | 14.29 | 20.60 | 0.889 | 0.221 |
| 60-min glucose | 24.50 | 33.30 | 0.709 | 0.184 |

Ensemble uncertainty was positively associated with actual forecast error. At 30 minutes, the median absolute error increased from 7.23 mg/dL in the lowest-uncertainty bin to 14.82 mg/dL in the highest-uncertainty bin. At 60 minutes, it increased from 14.93 mg/dL to 26.19 mg/dL.

This suggests that ensemble disagreement provides a useful, though imperfect, signal for identifying less reliable glucose forecasts.

## Selective Prediction

Selective prediction evaluates whether the model can improve reliability by abstaining from forecasts with high ensemble uncertainty. Predictions were ranked by ensemble standard deviation, and performance was recomputed after keeping only the most confident predictions.

| Target | Coverage | RMSE |
|---|---:|---:|
| 30-min glucose | 100% | 20.59 |
| 30-min glucose | 80% | 17.98 |
| 30-min glucose | 60% | 16.85 |
| 30-min glucose | 40% | 15.85 |
| 30-min glucose | 30% | 15.55 |
| 60-min glucose | 100% | 33.30 |
| 60-min glucose | 80% | 30.43 |
| 60-min glucose | 60% | 28.70 |
| 60-min glucose | 40% | 27.64 |
| 60-min glucose | 30% | 27.68 |

As coverage decreases, RMSE generally improves on the retained predictions. This suggests that ensemble uncertainty can support a practical trust mechanism: the model can provide forecasts when confidence is higher and flag uncertain cases for caution or further verification.

## Reliability Analyses

- Error vs uncertainty
- Selective prediction / abstention curves
- Patient-level error analysis
- Calibration for hypo/hyperglycemia risk
- Feature masking sensitivity analysis
- Uncertainty attribution for unreliable forecasts

## Disclaimer

This project is for research and educational purposes only. It is not intended for medical decision-making.