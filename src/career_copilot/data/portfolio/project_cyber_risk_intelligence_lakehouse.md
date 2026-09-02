---
id: project_cyber_risk_intelligence_lakehouse
title: "Project: Cyber Risk Intelligence Lakehouse"
tags: [project, pyspark, dbt, lakehouse, ml, shap, rag, terraform, aws, security]
repo: momo840505/cyber-risk-intelligence-lakehouse
---

# Cyber Risk Intelligence Lakehouse

End-to-end cyber risk intelligence lakehouse. Used PySpark to integrate three public threat
intelligence sources — CISA (Known Exploited Vulnerabilities), EPSS (Exploit Prediction
Scoring System), and NVD (National Vulnerability Database) — into a single pipeline, built
with dbt transformations on a lakehouse architecture, and trained a vulnerability
risk-prioritization classifier (with SHAP for explainability).

During project review, personally caught and fixed a target-variable leakage defect in the
model, ensuring the reported accuracy was trustworthy rather than inflated. This is a strong
"caught my own mistake through rigor" story for interviews.

Also built a RAG-based remediation-advice assistant on top of the risk data, plus a Streamlit
dashboard. Both are deployed live. Infrastructure is defined as code with Terraform and
deployed on AWS.

## Stack
Python, PySpark, dbt, machine learning, SHAP, FastAPI, Streamlit, Terraform, AWS.

## What this demonstrates
Data engineering at scale (PySpark + dbt + lakehouse), ML rigor (catching and fixing data
leakage before it shipped), infrastructure-as-code, and an early/simple RAG application —
which Career Copilot goes much deeper on (multi-step agent reasoning, citation enforcement,
evals), rather than a single-shot retrieval-then-answer assistant.
