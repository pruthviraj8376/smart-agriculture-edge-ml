"""
Central configuration for the Smart Agriculture Edge-ML project.
Change values here instead of editing the individual scripts.
"""
from pathlib import Path

# ---------- Folders ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DATASET_PATH = DATA_DIR / "sensor_data.csv"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
MODEL_NAME = "irrigation_predictor"

for _folder in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# ---------- Dataset ----------
RANDOM_SEED = 42
NUM_DEVICES = 3                 # number of simulated soil sensor nodes (fields)
DATASET_HOURS = 24 * 90         # 90 days of hourly readings per device
DATASET_START = "2026-01-01 00:00:00"

FEATURES = ["soil_moisture", "air_temp", "humidity", "light_lux", "rain_mm"]
TARGET = "irrigation_needed"

# A field needs irrigation when soil moisture drops below this % (used to label data)
IRRIGATION_THRESHOLD = 30.0

# ---------- Edge node ----------
ALERT_PROBABILITY = 0.60        # switch irrigation ON when P(needed) >= this value
SENSOR_INTERVAL_SEC = 1.0       # real-time delay between simulated readings (lower = faster demo)

# Valid physical ranges used by the edge node to reject faulty sensor data
VALID_RANGES = {
    "soil_moisture": (0.0, 100.0),
    "air_temp": (-10.0, 55.0),
    "humidity": (0.0, 100.0),
    "light_lux": (0.0, 120000.0),
    "rain_mm": (0.0, 200.0),
}
