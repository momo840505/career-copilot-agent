---
id: project_retail_demand_forecasting
title: "Project: Retail Demand Forecasting & Replenishment System"
tags: [project, ml, time-series, xgboost, deployment, aws, tableau, ab-testing]
repo: momo840505/retail-demand-forecasting
live_demo: https://retail-demand-forecasting-momo.streamlit.app/
---

# Retail Demand Forecasting & Replenishment System

Built a leakage-aware retail demand forecasting model (XGBoost) with time-series backtesting
across 4 rolling folds. Achieved a WAPE improvement of 24.5% over the baseline model.

Containerized the forecasting service and deployed it live to AWS Elastic Beanstalk, exposing
a publicly accessible API. Built a Tableau Public executive-level dashboard summarizing
forecast performance and inventory implications for a non-technical (management) audience.

Wrote a promotion-effectiveness A/B test experiment design document, including sample-size
and statistical power calculations.

## Stack
Python, XGBoost, FastAPI, Docker, AWS Elastic Beanstalk, Tableau (Public).

## What this demonstrates
End-to-end ML: careful validation methodology (avoiding data leakage in time-series CV),
real cloud deployment (not just a notebook), and translating model output into a
business-facing decision tool. Also demonstrates experimental design / causal-inference
adjacent skills (A/B test planning) beyond pure predictive modeling.
