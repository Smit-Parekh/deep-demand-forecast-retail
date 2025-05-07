# Deep Learning-Enhanced Demand Forecasting & Inventory Optimization (Retail/FMCG)

**Project Status:** [Active Development | Proof-of-Concept | Maintained]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## Overview

This repository contains the code and infrastructure definitions for an end-to-end demand forecasting system designed for Retail/FMCG clients. It addresses the limitations of traditional statistical models (ARIMA, ETS) by leveraging Deep Learning, specifically Temporal Fusion Transformers (TFTs), known for their ability to capture complex non-linearities, handle diverse inputs (static and dynamic), and provide interpretability.

The system integrates various data sources (sales history, promotions, holidays, macroeconomic indicators, web trends) and features a robust MLOps pipeline built on Google Cloud Vertex AI for automated retraining, hyperparameter optimization, deployment, and monitoring. The ultimate goal is to improve forecast accuracy (measured by WMAPE), enabling better inventory management within supply chains, reducing holding costs and stockouts.

## Key Features

*   **Advanced Forecasting Model:** Utilizes Temporal Fusion Transformers (TFT) via the `pytorch-forecasting` library.
*   **Rich Feature Engineering:** Incorporates time-varying known inputs (promotions, holidays), time-varying unknown inputs (sales lags, web trends), static categorical features (SKU info), and potentially macroeconomic data.
*   **MLOps on Vertex AI:**
    *   **Vertex AI Pipelines (KFP):** Orchestrates the end-to-end ML workflow (Data Validation -> Feature Engineering -> Training -> Evaluation -> Model Registration -> Deployment).
    *   **Vertex AI Training:** Enables distributed training for deep learning models.
    *   **Vertex AI Prediction:** Serves forecasts via scalable endpoints.
    *   **Vertex AI Feature Store:** Manages and serves features for training and prediction consistency.
    *   **Vertex AI Vizier:** Used for hyperparameter optimization.
*   **Explainable AI (XAI):** Leverages TFT's built-in attention mechanisms and SHAP integration for forecast interpretability.
*   **Data Integration:** Designed to work with data warehouses like Snowflake (simulated via CSV loading in this repo).
*   **Experiment Tracking:** Integrates with MLflow for tracking parameters, metrics, and model artifacts.
*   **API for Forecasts:** Includes a FastAPI endpoint for serving predictions on demand.
*   **Testing:** Comprehensive unit and integration tests using `pytest`.
*   **Containerization:** Uses Docker for consistent environments across development and deployment.
*   **CI/CD:** Basic GitHub Actions workflow for automated linting and testing.

## Problem Statement Addressed

Standard statistical forecasting models often fail to capture the complex dynamics impacting demand in the FMCG/Retail sector (e.g., non-linear responses to promotions, external event impacts). This leads to inaccurate forecasts and suboptimal inventory levels within the supply chain, resulting in increased costs (holding, stockouts) and potentially impacting client service levels.

## Solution & Differentiation

This project implements a Deep Learning approach (TFT) that demonstrably improves forecast accuracy (target: 15-20% WMAPE reduction) by modeling complex patterns and incorporating a wider range of influential features. The key differentiator lies in the combination of the advanced model with a production-grade MLOps pipeline on GCP, ensuring scalability, reproducibility, continuous improvement, and explainability crucial for operational adoption.

## Tech Stack

*   **Core:** Python 
*   **ML/DL:** PyTorch, PyTorch Lightning, pytorch-forecasting (for TFT)
*   **Data Handling:** Pandas, NumPy
*   **MLOps & Orchestration:** Google Cloud Vertex AI (Pipelines, Training, Prediction, Feature Store), MLflow, Docker
*   **Data Storage:** Snowflake, Google Cloud Storage (for artifacts)
*   **API:** FastAPI, Uvicorn
*   **Testing:** Pytest
*   **CI/CD:** GitHub Actions
*   **Explainability:** SHAP, Matplotlib, Seaborn
*   **Scheduling:** Cloud Scheduler (to trigger Vertex Pipelines)
*   **Visualization (External):** Power BI (consuming forecasts via API or direct DB connection)
