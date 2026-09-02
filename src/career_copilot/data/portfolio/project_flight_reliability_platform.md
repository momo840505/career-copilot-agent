---
id: project_flight_reliability_platform
title: "Project: Flight Reliability Platform — US Flight Data Engineering"
tags: [project, data-engineering, postgresql, star-schema, powerbi, data-quality]
repo: momo840505/flight-reliability-platform
---

# Flight Reliability Platform — US On-Time Performance Data Engineering

Processed official US Department of Transportation flight data, over 540,000 flight records
in a single month, cleaning and converting to Parquet format (roughly an 86% size reduction).

Designed a PostgreSQL star-schema data warehouse, with data-quality validation at every layer
(structural checks, type checks, business-logic checks, duplicate detection, foreign-key
integrity). Built SQL analytical views plus an interactive Power BI dashboard answering real
business questions such as on-time-performance rankings by airline and by airport.

## Stack
Python, PostgreSQL, SQL, Power BI, data engineering.

## What this demonstrates
Classic data-engineering rigor at meaningful scale: ETL, dimensional modeling (star schema),
and — importantly — data-quality validation built into every pipeline layer, not bolted on
after the fact. Career Copilot's "critic" node and eval harness are the same instinct
(validate before you trust the output) applied to an LLM pipeline instead of a SQL pipeline.
