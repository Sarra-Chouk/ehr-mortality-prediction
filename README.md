# EHR Mortality Prediction

This project develops an end-to-end machine learning pipeline for predicting patient mortality risk from structured electronic health record data. It combines hospital and ICU source tables into a single encounter-level feature set, prepares a stable model-ready dataset, trains and calibrates predictive models, and produces outputs that can support downstream scoring and dashboarding workflows.

The core objective is to estimate mortality risk from routinely collected clinical and operational variables. The feature engineering pipeline brings together demographic information, admission context, prior utilization patterns, diagnosis and procedure history, medication burden, transfer and ICU history, infection-related signals, laboratory summaries, and recent vital-sign features. The result is a standardized dataset designed for both experimentation and deployment.

## Project Scope

The repository covers the full modeling lifecycle:

- ingestion and consolidation of raw EHR tables into a master analytical table
- preprocessing, schema validation, categorical encoding, and model-ready dataset generation
- patient-level train, test, and deployment splits with leakage-aware temporal imputation
- baseline and advanced model experimentation in notebooks
- calibrated XGBoost model packaging for batch inference
- prediction export for deployment and dashboard-ready reporting
- explainability and bias analysis artifacts for model interpretation

## Modeling Approach

The production-oriented modeling workflow centers on a calibrated XGBoost classifier. The project includes utilities to train the model, estimate class imbalance weights, calibrate predicted probabilities, and package the final model together with its expected input schema and metadata. This supports more reliable risk scoring and reproducible downstream use.

Alongside the packaged model workflow, the notebooks document broader experimentation across multiple approaches, including traditional baseline models, neural network modeling, explainability analysis with SHAP, subgroup bias analysis, and comparison against alternative tabular modeling strategies.

## Repository Structure

- `src/data/`: data assembly, feature engineering, preprocessing, and split generation
- `src/models/`: model packaging and dashboard-oriented prediction utilities
- `deployment/`: batch scoring script and packaged model outputs
- `notebooks/`: exploratory analysis, model development, explainability, and fairness analysis
- `artifacts/`: generated model artifacts and analytical outputs

## Outputs

The project produces several deployment- and analysis-oriented outputs:

- a master feature table derived from raw EHR sources
- a validated model-ready dataset with a stable schema
- train, test, and deployment data partitions
- packaged model files and prediction metadata
- scored prediction files for deployment use
- dashboard-ready prediction tables
- explainability visualizations and fairness assessment outputs

## Summary

This repository is an applied clinical machine learning project focused on transforming raw EHR data into an interpretable and deployment-ready mortality risk prediction pipeline. It is structured to support data preparation, model development, evaluation, packaging, and analytical reporting within a single codebase.
