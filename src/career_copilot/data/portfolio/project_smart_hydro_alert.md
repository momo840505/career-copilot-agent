---
id: project_smart_hydro_alert
title: "Project: Smart Hydro Alert — Real-time IoT Leak Detection"
tags: [project, iot, mqtt, websocket, fastapi, mongodb, react, realtime]
repo: momo840505/smart-hydro-alert
---

# Smart Hydro Alert — Real-time IoT Water-Waste & Leak Detection

Full-stack real-time IoT system (ESP32 sensors in the original design) built on an MQTT +
WebSocket architecture, where sensor anomalies reach the front-end dashboard within seconds.

Designed cross-sensor validation logic: a high-risk leak alert only fires when both water-flow
and localized water-contact sensors agree, reducing false positives from any single faulty
sensor. High-risk events automatically trigger Telegram notifications. Includes a software
simulator so the system can be demonstrated without physical hardware.

## Stack
Python, FastAPI, MongoDB, MQTT, React, IoT.

## What this demonstrates
Real-time/streaming system design, sensor-fusion-style logic to reduce false alarms
(conceptually similar to the "critic" self-check Career Copilot uses to reduce hallucinated
claims), full-stack development (React front end + FastAPI/MongoDB back end), and designing
for demoability (software simulator) when hardware isn't available.
