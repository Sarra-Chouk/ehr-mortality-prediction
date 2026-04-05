# EHR Mortality Prediction

This repository contains an end-to-end clinical machine learning workflow for predicting **in-hospital mortality risk** from structured electronic health record data. The project turns raw MIMIC-IV hospital data into an encounter-level feature table, evaluates multiple predictive modeling strategies, analyzes model behavior with explainability and fairness workflows, and packages the selected model for deployment-oriented scoring.

## Project Overview

| Area | Summary |
| --- | --- |
| **Business Problem** | Identify patients at elevated mortality risk early enough to support downstream monitoring, prioritization, and clinical decision support workflows. |
| **Objectives** | Build a leakage-aware mortality prediction pipeline, compare candidate models, select a production-ready approach, and generate interpretable outputs for deployment and dashboarding. |
| **Dataset** | Structured **MIMIC-IV** hospital and ICU data consolidated into an encounter-level master table with **546,028 rows** and **46 engineered columns**. |
| **Final Deployment Asset** | A packaged XGBoost-based inference pipeline with saved schema, metadata, threshold, and dashboard-ready scoring outputs. |

## Dataset

The project uses de-identified **MIMIC-IV** EHR data, primarily from the hospital and ICU domains. Raw source tables are consolidated into a master analytical table where each row represents a hospital admission. The engineered feature set combines:

- demographics and admission context
- emergency department timing and entry pattern
- prior utilization and length-of-stay history
- diagnosis and procedure burden
- DRG severity and mortality indicators
- laboratory abnormality summaries
- medication exposure history
- transfer, ICU, infection, and resistance history
- recent OMR and vital-sign derived features

The final processed data is split at the **patient level** into train, test, and deployment partitions to prevent cross-split leakage:

- **Train:** 380,461 rows
- **Test:** 83,356 rows
- **Deployment:** 82,211 rows

## Analytical Workflow

### 1. Feature Engineering

Raw MIMIC-IV hospital tables are consolidated into an admission-level master table through a leakage-aware feature engineering pipeline. This stage derives historical and encounter-level predictors from diagnoses, procedures, DRG codes, laboratory events, prescriptions, transfers, microbiology records, and outpatient measurements. The resulting feature space captures both current admission context and prior patient history.

### 2. Data Preprocessing

The engineered master table is validated against an expected schema and transformed into a stable model-ready dataset. This stage standardizes missing values, enforces data types, handles categorical levels, applies one-hot encoding, and preserves a consistent feature schema for training and deployment. Patient-level train, test, and deployment splits are then created to avoid cross-patient leakage, with a separate imputed split path for models that do not support missing values natively.

### 3. Exploratory Data Analysis

EDA was used to profile class imbalance, missingness, feature distributions, subgroup differences, and clinically relevant relationships before modeling. The notebooks cover both summary-level inspection and visual analysis of demographic, operational, laboratory, medication, ICU, and infection-related signals.

### 4. Predictive Modeling Experiments

The modeling workflow compares multiple approaches for an imbalanced binary classification problem. Experiments include:

- baseline machine learning models such as Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, Naive Bayes, XGBoost, and LightGBM
- class-imbalance handling through weighting and threshold tuning
- SMOTETomek-based resampling experiments
- random-search XGBoost model selection using validation **PR-AUC** and **ROC-AUC**
- additional neural-network and tabular-model experiments, including **MLP** and **TabICLv2**

The selected production model is the saved **XGBoost** artifact under [`artifacts/xgboost/`](artifacts/xgboost), which is reused for packaging and deployment scoring.

### 5. SHAP Explainability

The repository includes SHAP-based explainability analysis for the final XGBoost model. This includes:

- global feature-importance views
- distributional SHAP plots across the test set
- local case-level explanations for representative predictions

This layer is intended to make the model behavior more interpretable and defensible beyond raw performance metrics alone.

### 6. Bias Analysis

Model behavior is also evaluated across key demographic subgroups to surface performance disparities. The bias-analysis workflow reports subgroup metrics across:

- gender
- age group
- race group

This supports a more complete assessment of deployment risk by examining fairness gaps in recall, precision, F1, ROC-AUC, and PR-AUC rather than relying only on aggregate performance.

## Repository Structure

- [`src/data/`](src/data): raw-table feature engineering, preprocessing, and split generation
- [`src/models/`](src/models): final model-specific logic for the selected XGBoost approach
- [`src/deployment/`](src/deployment): packaged inference pipeline and dashboard table generation
- [`deployment/`](deployment): deployment scoring script and packaged model files
- [`notebooks/`](notebooks): EDA, modeling experiments, SHAP explainability, and bias analysis

## Summary

This project is an applied mortality-risk modeling pipeline built around structured EHR data, leakage-aware feature engineering, comparative model evaluation, and deployment-oriented inference. It is designed to support both technical experimentation and practical scoring workflows while maintaining interpretability through SHAP analysis and fairness checks across key patient subgroups.
