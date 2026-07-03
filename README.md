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

## Reliability Analyses

- Error vs uncertainty
- Selective prediction / abstention curves
- Patient-level error analysis
- Calibration for hypo/hyperglycemia risk
- Feature masking sensitivity analysis
- Uncertainty attribution for unreliable forecasts

## Disclaimer

This project is for research and educational purposes only. It is not intended for medical decision-making.