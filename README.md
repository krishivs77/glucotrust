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

## Reliability Analyses

- Error vs uncertainty
- Selective prediction / abstention curves
- Patient-level error analysis
- Calibration for hypo/hyperglycemia risk
- Feature masking sensitivity analysis
- Uncertainty attribution for unreliable forecasts

## Disclaimer

This project is for research and educational purposes only. It is not intended for medical decision-making.