---
title: "Real-Time Machine Learning at the IoT Edge for Smart Agriculture"
subtitle: "Irrigation Need Prediction using Simulated Sensor Data, Model Registry, Edge Inference and a Live Dashboard"
author: "[Student Name] · [Roll No.] · [Department / Course: AEAL]"
date: "[Submission Date]"
---

# Abstract

Irrigation is one of the largest uses of fresh water in agriculture, and over-watering or late watering both reduce yield. This project builds an end-to-end, edge-deployed machine learning system that predicts whether a field needs irrigation in the next hour. A network of simulated soil sensors streams temperature, humidity, light, rainfall and soil moisture readings to an edge node. The edge node validates each reading, runs a Random Forest classifier obtained from a local model registry, decides locally whether to switch irrigation on, and records the result. A Streamlit dashboard displays the predictions, decisions, latency and model performance in real time. The model reached a ROC-AUC of 0.974 and an F1-score of 0.72 on a held-out test set, and the live edge pipeline achieved 93.5% accuracy against ground truth with an average inference latency of 17 ms per reading.

**Keywords:** Internet of Things (IoT), Edge computing, Machine learning, Random Forest, Smart agriculture, Model registry, Real-time dashboard

# 1. Introduction

Smart agriculture uses sensors, connectivity and data analysis to make farming more efficient. Soil moisture is a key variable: when it falls below the level a crop needs, the field must be irrigated, and when it is above that level, watering wastes water and energy. Sending every sensor reading to a cloud server for a decision adds latency and depends on connectivity, which is often poor in fields. Processing data on an *edge* device close to the sensors avoids these problems.

This project combines three areas taught in the course: IoT data acquisition, machine learning model development and deployment, and real-time data visualisation. The result is a complete working prototype that can be run on a single laptop.

## 1.1 Problem statement

Given the most recent readings from a soil-sensor node (soil moisture, air temperature, humidity, light intensity and rainfall), predict whether the field will need irrigation in the next hour, and let an edge node take an irrigation decision in real time.

## 1.2 Objectives

1. Prepare a suitable dataset for the smart-agriculture domain.
2. Train a machine learning model and evaluate it with appropriate metrics.
3. Register and manage model versions with stages (Staging, Production, Archived).
4. Integrate the model into an edge node that validates incoming IoT data and makes decisions locally.
5. Simulate incoming IoT sensor data in real time.
6. Generate predictions and display the results on a live dashboard.

# 2. Literature and Background

Random Forest is an ensemble method that builds many decision trees on random subsets of the data and combines their votes. It handles mixed sensor features well, is robust to noise, and provides feature-importance scores that help explain the model. Pedregosa et al. (2011) describe the scikit-learn library used here. Edge computing moves computation to devices near the data source to reduce latency and bandwidth use. Model registries record model versions, their metrics and their deployment stage so that the correct model is always used. Streamlit is a Python framework for building interactive data applications.

# 3. System Architecture

The system consists of six components connected in a pipeline:

| Component | File | Responsibility |
|---|---|---|
| Dataset generator | `generate_dataset.py` | creates realistic hourly sensor data for three fields |
| Trainer | `train_model.py` | prepares data, trains and evaluates the model |
| Model registry | `model_registry.py` | stores versions, metrics and stages |
| Sensor simulator | `iot_simulator.py` | streams readings one by one, with occasional faults |
| Edge node | `edge_node.py` | validates, predicts, decides and logs |
| Dashboard | `dashboard.py` | shows results from the log file |

The simulator and the edge node run as separate threads connected by a queue, which imitates a message broker such as MQTT. The edge node writes results to a CSV file, and the dashboard reads that file. This decoupling means the dashboard can be started, stopped or replaced without affecting inference.

# 4. Dataset

## 4.1 Data generation

No public dataset matched the required combination of sensors, so a physically motivated simulator was written (`generate_dataset.py`). For each of three fields, 90 days of hourly readings were generated (6,480 rows in total):

- **Air temperature** follows a seasonal cycle and a daily cycle, with random noise.
- **Light intensity** is zero at night and peaks at midday.
- **Rainfall** occurs in random storm events.
- **Humidity** is higher when the air is cooler and after rain.
- **Soil moisture** changes hour by hour: it falls through evaporation (faster when hot, bright and dry) and rises after rain. A farmer waters the field when it becomes dry, with a probability of 25% per hour below the threshold, which gives realistic dry–wet cycles.
- **Sensor noise and faults** are added: small Gaussian measurement noise, and 1% of soil readings are missing.

## 4.2 Features and label

| Feature | Unit | Description |
|---|---|---|
| `soil_moisture` | % | volumetric soil water content measured by the sensor |
| `air_temp` | °C | air temperature |
| `humidity` | % | relative humidity |
| `light_lux` | lux | ambient light |
| `rain_mm` | mm | rainfall in the hour |

**Target:** `irrigation_needed` = 1 if soil moisture in the *next* hour is below 30%, otherwise 0. Irrigation was needed in **10.7%** of readings, so the classes are imbalanced, and the model uses balanced class weights.

## 4.3 Data cleaning

Rows with missing soil readings (sensor dropouts) are removed before training. At run time the edge node performs the same kind of check, plus range validation (for example, air temperature must be between −10 °C and 55 °C). Readings outside these physical limits are rejected and logged with the reason.

# 5. Methodology

## 5.1 Model training

- **Split:** 80% training (5,127 rows) and 20% testing (1,282 rows), stratified on the target so both sets have the same class ratio.
- **Model:** Random Forest Classifier with 200 trees, maximum depth 10, minimum 5 samples per leaf, balanced class weights and a fixed random seed for repeatability.
- **Metrics:** accuracy, precision, recall, F1-score, ROC-AUC and the confusion matrix. For an irrigation system, **recall** matters most, because a missed irrigation event (a field left dry) is more costly than an unnecessary watering.

## 5.2 Model registry

Each training run is saved as a versioned file (for example `irrigation_predictor_v1.joblib`) and its metadata (metrics, features, hyperparameters, date and stage) is written to `model_registry.json`. Promoting a new version to *Production* automatically archives the old one. The edge node always loads the current Production model, so a retrained model can be deployed by running one command.

## 5.3 Edge inference

For every reading the edge node:

1. checks that all required features are present and inside their valid physical ranges;
2. computes the probability that irrigation is needed, using the Production model;
3. decides an action: **IRRIGATION_ON** if the probability is at least 0.60, **IRRIGATION_OFF** if it is below 0.30, otherwise **MONITOR**;
4. measures the inference latency;
5. appends the result to `results/predictions.csv`.

# 6. Results

## 6.1 Offline model performance (held-out test set)

| Metric | Value |
|---|---|
| Accuracy | 0.924 |
| Precision | 0.594 |
| Recall | 0.927 |
| F1-score | 0.724 |
| ROC-AUC | 0.974 |

Confusion matrix (rows = actual, columns = predicted):

| | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | 1,058 (true negatives) | 87 (false positives) |
| **Actual: Yes** | 10 (false negatives) | 127 (true positives) |

The model detects **93% of the situations that need irrigation** (recall 0.927) and misses only 10 of 137 such cases. The 87 false alarms mean some fields were flagged as needing water when they did not; for a farm this is a small cost compared with a dry crop.

## 6.2 Feature importance

| Feature | Importance |
|---|---|
| Soil moisture | 0.886 |
| Air temperature | 0.053 |
| Humidity | 0.039 |
| Light intensity | 0.021 |
| Rainfall | 0.001 |

Soil moisture dominates, which agrees with the agronomic meaning of the target. Temperature and humidity contribute through their effect on evaporation.

## 6.3 Live edge pipeline

A run of the pipeline on 1,500 streamed readings (three fields) gave:

| Measure | Value |
|---|---|
| Readings processed by the model | 1,455 |
| Readings rejected by validation | 45 (faulty values and missing data) |
| Live accuracy against ground truth | 93.5% |
| Actions: IRRIGATION_ON / MONITOR / IRRIGATION_OFF | 191 / 58 / 1,206 |
| Average inference latency | 17.4 ms per reading |

The 3% of faulty readings were all caught by the validation step, so none reached the model. The latency is well within what is needed for field-scale decisions, which are typically made every few minutes.

## 6.4 Dashboard

The Streamlit dashboard shows: KPI cards (processed readings, rejected readings, live accuracy, irrigation actions, average latency); soil-moisture trends per field with the irrigation threshold; the distribution of edge decisions; the model's probability over time; a predicted-versus-actual confusion matrix; the feature-importance chart; the latest predictions; a log of rejected readings; the model registry; and a CSV download. Auto-refresh updates the view every few seconds while the pipeline runs.

# 7. Testing and Validation

| Test | Method | Result |
|---|---|---|
| Dataset generation | script runs; row count and label balance checked | 6,480 rows, 10.7% positive |
| Model training | script runs; metrics computed on held-out data | ROC-AUC 0.974 |
| Model registry | register, promote, reload model | version 1 saved and reloaded as Production |
| Data validation | faulty readings injected (air temperature 999 °C, missing soil value) | all 45 faulty readings rejected with reasons |
| End-to-end pipeline | simulator → queue → edge node → CSV | 1,500 readings processed without error |
| Dashboard | automated page load with no exceptions | all KPIs, charts and tables rendered |

# 8. Discussion

The results show that a lightweight Random Forest model can run on an edge device with millisecond-level latency and produce useful irrigation decisions. Using a registry and a validation step made the system more reliable than a model script alone.

The project also has limitations that should be stated honestly:

1. **The data is simulated.** The sensor behaviour follows physical rules that were written by the author, so the accuracy measured here is an indication of the method rather than of a real farm. Real sensors have calibration drift, weather forecasts are uncertain and soil types differ.
2. **The label is close to a threshold on the input.** Because the target is "soil moisture below 30% in the next hour", soil moisture is by far the strongest feature. A useful extension is to forecast the moisture value itself, or to include weather forecasts.
3. **The registry is local.** It stores versions in a JSON file. Production systems would use a tool such as MLflow with a central server.
4. **Single edge node.** The three fields share one process. A real deployment would run one edge node per farm or per gateway, with a message broker such as MQTT.
5. **Actions are simulated.** The IRRIGATION_ON decision is written to a log and not sent to a pump or valve.

# 9. Future Work

- Connect to real sensors over MQTT or a serial port (for example a Raspberry Pi with capacitive soil-moisture probes).
- Add a forecasting model (LSTM or gradient boosting on lagged features) to predict soil moisture several hours ahead.
- Use live weather forecasts to avoid irrigating before rain.
- Register models in MLflow and add automatic retraining when accuracy drops.
- Add alerts by email or SMS when a field is dry for a long period.
- Compare Random Forest with other models (logistic regression, gradient boosting, a small neural network) on latency, memory and accuracy.

# 10. Conclusion

This project delivered a complete smart-agriculture system that follows the full machine learning life cycle: dataset preparation, training, registration, edge integration, simulated IoT streaming, real-time prediction and visualisation. The Random Forest model achieved a ROC-AUC of 0.974 and detected 93% of the field-dry situations on held-out data, and the edge pipeline processed each reading in about 17 ms with validation of faulty sensor data. The work shows how edge intelligence can reduce water use in agriculture, and it provides a clear base for connecting to real hardware in future work.

# References

1. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
3. Shi, W., Cao, J., Zhang, Q., Li, Y., & Xu, L. (2016). Edge Computing: Vision and Challenges. *IEEE Internet of Things Journal*, 3(5), 637–646.
4. Streamlit Inc. Streamlit documentation. https://docs.streamlit.io
5. Plotly Technologies Inc. Plotly Python documentation. https://plotly.com/python/

# Appendix A – How to Run

1. `python generate_dataset.py`
2. `python train_model.py`
3. `python run_pipeline.py --rows 300 --interval 0.5`
4. `streamlit run dashboard.py` (in a second terminal)

Full setup instructions for VS Code are in `README.md`.
