"""
Step 3 - Simulated IoT sensor network.

Each simulated device (soil sensor node) sends one reading per interval into a shared queue,
the same way real sensors would publish messages to an edge gateway.
Readings are replayed from data/sensor_data.csv so the simulation is realistic and repeatable.
Occasionally a device sends a faulty reading (out-of-range value) to show data validation.

Used by run_pipeline.py - you normally do not run this file on its own.
"""
import queue
import random
import threading
import time

import pandas as pd

import config


class SensorSimulator(threading.Thread):
    def __init__(self, out_queue: queue.Queue, interval_sec: float = config.SENSOR_INTERVAL_SEC,
                 max_rows: int | None = None, fault_rate: float = 0.02, seed: int = config.RANDOM_SEED):
        super().__init__(daemon=True, name="sensor-simulator")
        self.out_queue = out_queue
        self.interval = interval_sec
        self.max_rows = max_rows
        self.fault_rate = fault_rate
        self.rng = random.Random(seed)
        self.stop_event = threading.Event()
        self.sent = 0

    def _load_stream(self) -> pd.DataFrame:
        df = pd.read_csv(config.DATASET_PATH, parse_dates=["timestamp"])
        # Interleave devices in time order so each "tick" contains one reading per field
        return df.sort_values(["timestamp", "device_id"]).reset_index(drop=True)

    def run(self):
        stream = self._load_stream()
        if self.max_rows:
            stream = stream.head(self.max_rows)

        for _, row in stream.iterrows():
            if self.stop_event.is_set():
                break
            reading = {
                "timestamp": row["timestamp"].isoformat(),
                "device_id": row["device_id"],
                "soil_moisture": row["soil_moisture"],
                "air_temp": row["air_temp"],
                "humidity": row["humidity"],
                "light_lux": row["light_lux"],
                "rain_mm": row["rain_mm"],
                "actual_irrigation_needed": int(row[config.TARGET]),  # ground truth for evaluation
            }
            # Simulate a faulty sensor now and then
            if self.rng.random() < self.fault_rate:
                reading["air_temp"] = 999.0  # impossible value -> edge node must reject it
            self.out_queue.put(reading)
            self.sent += 1
            time.sleep(self.interval)

        self.out_queue.put(None)  # end-of-stream marker

    def stop(self):
        self.stop_event.set()
