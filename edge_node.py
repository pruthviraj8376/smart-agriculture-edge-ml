"""
Step 4 - Edge node: real-time inference close to the sensors.

The edge node:
  1. receives each sensor reading from the IoT queue,
  2. validates it (missing values and out-of-range values are rejected),
  3. runs the Production model from the registry to get P(irrigation needed),
  4. decides the local irrigation action (ON / OFF) without needing the cloud,
  5. measures the inference latency and appends the result to results/predictions.csv.

The dashboard reads results/predictions.csv, so the two are fully decoupled.
"""
import csv
import math
import queue
import time
from datetime import datetime

import pandas as pd

import config
import model_registry

RESULT_COLUMNS = [
    "received_at", "timestamp", "device_id", "soil_moisture", "air_temp", "humidity",
    "light_lux", "rain_mm", "probability", "predicted_irrigation", "actual_irrigation_needed",
    "action", "latency_ms", "status", "reason",
]


class EdgeNode:
    def __init__(self, in_queue: queue.Queue, output_path=config.PREDICTIONS_PATH,
                 alert_probability: float = config.ALERT_PROBABILITY):
        self.in_queue = in_queue
        self.output_path = output_path
        self.alert_probability = alert_probability
        self.model, self.model_info = model_registry.get_production_model()
        self.counts = {"processed": 0, "rejected": 0}
        self._prepare_output()

    # ---------- setup ----------
    def _prepare_output(self):
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(RESULT_COLUMNS)

    # ---------- validation ----------
    @staticmethod
    def validate(reading: dict) -> str | None:
        """Return None if valid, otherwise a short reason string."""
        for feature in config.FEATURES:
            value = reading.get(feature)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return f"missing {feature}"
            low, high = config.VALID_RANGES[feature]
            if not (low <= float(value) <= high):
                return f"{feature} out of range ({value})"
        return None

    # ---------- inference ----------
    def predict(self, reading: dict) -> tuple[float, int]:
        features = pd.DataFrame([[reading[f] for f in config.FEATURES]], columns=config.FEATURES)
        probability = float(self.model.predict_proba(features)[0][1])
        predicted = int(probability >= 0.5)
        return probability, predicted

    def decide_action(self, probability: float) -> str:
        if probability >= self.alert_probability:
            return "IRRIGATION_ON"
        if probability < 0.3:
            return "IRRIGATION_OFF"
        return "MONITOR"

    # ---------- main loop ----------
    def _log(self, row: dict):
        with open(self.output_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=RESULT_COLUMNS).writerow(row)

    def process(self, reading: dict):
        started = time.perf_counter()
        base = {
            "received_at": datetime.now().isoformat(timespec="seconds"),
            "timestamp": reading.get("timestamp"),
            "device_id": reading.get("device_id"),
            "soil_moisture": reading.get("soil_moisture"),
            "air_temp": reading.get("air_temp"),
            "humidity": reading.get("humidity"),
            "light_lux": reading.get("light_lux"),
            "rain_mm": reading.get("rain_mm"),
            "actual_irrigation_needed": reading.get("actual_irrigation_needed"),
        }

        reason = self.validate(reading)
        if reason:
            self.counts["rejected"] += 1
            self._log({**base, "probability": None, "predicted_irrigation": None,
                       "action": "NONE", "latency_ms": None, "status": "REJECTED", "reason": reason})
            return

        probability, predicted = self.predict(reading)
        latency_ms = (time.perf_counter() - started) * 1000
        self.counts["processed"] += 1
        self._log({**base, "probability": round(probability, 4), "predicted_irrigation": predicted,
                   "action": self.decide_action(probability), "latency_ms": round(latency_ms, 3),
                   "status": "OK", "reason": ""})

    def run(self, stop_after_end_marker: bool = True):
        print(f"[edge] Loaded model v{self.model_info['version']} "
              f"(F1={self.model_info['metrics']['f1']}) - waiting for sensor data...")
        while True:
            try:
                reading = self.in_queue.get(timeout=30)
            except queue.Empty:
                print("[edge] No data for 30 s, stopping.")
                break
            if reading is None:  # simulator finished
                if stop_after_end_marker:
                    break
                continue
            self.process(reading)
        print(f"[edge] Finished. processed={self.counts['processed']} rejected={self.counts['rejected']}")
