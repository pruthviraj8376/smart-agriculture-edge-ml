# 🌱 Smart Agriculture – Real-Time ML at the IoT Edge

A complete, runnable project that predicts **whether a field needs irrigation** from simulated IoT soil sensors, using a Machine Learning model deployed on an **edge node** and shown on a **live dashboard**. Everything runs locally on your computer.

---

## 1. What the project does

```
 Simulated IoT sensors        Edge node (ML)                 Dashboard
 (3 fields, 1 reading/s)  ──▶ validate → predict → act  ──▶ results/predictions.csv ──▶ Streamlit
 soil, temp, humidity,        Random Forest model            (live charts, KPIs,
 light, rain                  from model registry            tables, model info)
```

| Requirement | Where it is done |
|---|---|
| Select a domain (Smart Agriculture) | whole project |
| Prepare a dataset | `generate_dataset.py` → `data/sensor_data.csv` |
| Train an ML model | `train_model.py` (Random Forest) |
| Manage / register the model | `model_registry.py` → `models/model_registry.json` (versions, stages, metrics) |
| Integrate model with an IoT / edge node | `edge_node.py` (validation, inference, local ON/OFF decision) |
| Simulate incoming IoT data | `iot_simulator.py` (threaded sensor simulator + faults) |
| Generate results using the model | `results/predictions.csv` |
| Display results on a dashboard | `dashboard.py` (Streamlit + Plotly) |

---

## 2. Folder structure

```
smart_agriculture_edge_ml/
├── README.md                 ← this guide
├── PROJECT_REPORT.md         ← project report (also available as .docx)
├── requirements.txt          ← Python libraries
├── config.py                 ← all settings (thresholds, paths, seeds)
├── generate_dataset.py       ← Step 1: create simulated IoT dataset
├── train_model.py            ← Step 2: train + register the model
├── model_registry.py         ← model versioning / stages
├── iot_simulator.py          ← Step 3: simulated sensor network
├── edge_node.py              ← Step 4: real-time edge inference
├── run_pipeline.py           ← Step 5: runs simulator + edge node together
├── dashboard.py              ← Step 6: live dashboard
├── data/                     ← dataset (created automatically)
├── models/                   ← trained model files + registry (created automatically)
└── results/                  ← predictions and feature importance (created automatically)
```

---

## 3. Setup in VS Code (step by step)

### Step 3.1 – Install the tools
1. Install **Python 3.10 or newer** from https://www.python.org/downloads/ (on Windows, tick **"Add python.exe to PATH"** during install).
2. Install **Visual Studio Code** from https://code.visualstudio.com/.
3. In VS Code, open the **Extensions** panel (Ctrl+Shift+X / Cmd+Shift+X) and install **Python** (by Microsoft).

### Step 3.2 – Open the project
1. Unzip the project folder, e.g. to `Documents/smart_agriculture_edge_ml`.
2. In VS Code: **File → Open Folder…** and select `smart_agriculture_edge_ml`.
3. Open the terminal: **Terminal → New Terminal** (Ctrl+` / Cmd+`).

### Step 3.3 – Create a virtual environment (a private Python for this project)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks scripts, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

When it works, the terminal line starts with `(.venv)`.

### Step 3.4 – Select the interpreter in VS Code
Press **Ctrl+Shift+P** (Cmd+Shift+P on Mac) → type **"Python: Select Interpreter"** → choose the one that shows **`.venv`**.

### Step 3.5 – Install the libraries
With `(.venv)` active, run:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | data handling and simulation |
| `scikit-learn` | Random Forest model and metrics |
| `joblib` | saving/loading the model file |
| `streamlit` | the web dashboard |
| `plotly` | interactive charts |

Check everything installed:
```bash
python -c "import pandas, sklearn, streamlit, plotly; print('All libraries OK')"
```

---

## 4. Run the project

Run these in order from the project folder (with `(.venv)` active).

### Step 4.1 – Generate the dataset
```bash
python generate_dataset.py
```
Creates `data/sensor_data.csv` (6,480 hourly readings from 3 fields over 90 days).

### Step 4.2 – Train and register the model
```bash
python train_model.py
```
Trains a Random Forest, prints accuracy / precision / recall / F1 / ROC-AUC, saves it to `models/irrigation_predictor_v1.joblib` and registers it as **Production**.

To see all registered versions:
```bash
python model_registry.py
```

### Step 4.3 – Run the real-time pipeline (sensors → edge → predictions)
```bash
python run_pipeline.py --rows 300 --interval 0.5
```
- `--rows` = how many sensor readings to stream
- `--interval` = seconds between readings (use `0` for instant processing)
- `--fault-rate` = share of faulty sensor readings to simulate (default 0.02)

The edge node prints a summary and writes every decision to `results/predictions.csv`.

### Step 4.4 – Open the dashboard
Open a **second terminal** in VS Code (the + button in the terminal panel), activate `.venv` again, and run:
```bash
streamlit run dashboard.py
```
Your browser opens automatically at **http://localhost:8501**. If not, open that link manually.

Keep `run_pipeline.py` running in the first terminal while you watch the dashboard update. To stop Streamlit press **Ctrl+C**.

> **Tip:** The project already contains a sample run (1,500 readings), so the dashboard shows results even before you run the pipeline.

---

## 5. How the dashboard works

| Section | Shows |
|---|---|
| KPI cards | readings processed, rejected sensor readings, live accuracy against ground truth, irrigation ON actions, average inference latency |
| Soil moisture by field | moisture over time with the irrigation threshold line (red dashed) |
| Edge decisions | count of IRRIGATION_ON / IRRIGATION_OFF / MONITOR actions |
| Model probability | P(irrigation needed) over time with the alert threshold |
| Predicted vs actual | confusion matrix |
| What the model relies on | feature importance |
| Latest predictions / rejected readings | tables of recent results |
| Model registry | versions, stage, metrics |

Use the **sidebar** to toggle auto-refresh, change the refresh speed and filter fields. You can download the predictions as CSV from the bottom of the page.

---

## 6. Key settings (edit `config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `IRRIGATION_THRESHOLD` | 30 % | soil moisture below this needs irrigation |
| `ALERT_PROBABILITY` | 0.60 | edge node turns irrigation ON at this probability |
| `SENSOR_INTERVAL_SEC` | 1.0 | default delay between simulated readings |
| `VALID_RANGES` | – | physical limits used to reject faulty sensor data |
| `RANDOM_SEED` | 42 | makes results repeatable |

After changing the dataset settings, run `generate_dataset.py` and `train_model.py` again.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | activate `.venv` and run `pip install -r requirements.txt` |
| `Dataset not found` | run `python generate_dataset.py` |
| `No Production model found` | run `python train_model.py` |
| Dashboard says "No results yet" | run `python run_pipeline.py` |
| `streamlit` not recognised | run it as `python -m streamlit run dashboard.py` |
| Port 8501 busy | `streamlit run dashboard.py --server.port 8502` |
| Charts look stale | turn on **Auto-refresh** in the sidebar |
| Wrong interpreter in VS Code | Ctrl+Shift+P → Python: Select Interpreter → `.venv` |

---

## 8. Optional extension – MLflow

The registry in `model_registry.py` is a simple local version of a model registry. For a full tool, you can install **MLflow** (`pip install mlflow`) and log the same metrics and model with `mlflow.sklearn.log_model(...)`. Mention this in the report as future work or as an enhancement.
