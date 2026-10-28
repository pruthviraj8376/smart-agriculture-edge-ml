"""
Step 1 - Generate a simulated IoT dataset for smart agriculture.

Each device (a soil sensor node in a field) produces hourly readings:
    soil_moisture (%), air_temp (C), humidity (%), light_lux, rain_mm

The soil moisture follows a realistic pattern:
  * it dries out faster on hot, dry, sunny days (evaporation),
  * it rises when it rains,
  * sensors add small random noise and occasionally drop readings.

Label: irrigation_needed = 1 if the NEXT hour's soil moisture is below the threshold.
Run:   python generate_dataset.py
"""
import numpy as np
import pandas as pd

import config


def simulate_device(device_id: str, hours: int, rng: np.random.Generator) -> pd.DataFrame:
    timestamps = pd.date_range(config.DATASET_START, periods=hours, freq="h")
    hour_of_day = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    # Seasonal + daily temperature cycle (degrees C)
    seasonal = 6 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    daily = 7 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    air_temp = 24 + seasonal + daily + rng.normal(0, 1.2, hours)

    # Light: zero at night, bright at midday
    light_lux = np.clip(np.sin(np.pi * (hour_of_day - 6) / 12), 0, None) * 85000
    light_lux = light_lux * rng.uniform(0.6, 1.0, hours)

    # Rain: occasional storms (about 1 rainy event every ~3 days)
    rain_mm = np.zeros(hours)
    rain_event = rng.random(hours) < 0.012
    rain_mm[rain_event] = rng.gamma(2.0, 2.5, rain_event.sum())

    # Humidity is higher when it is cooler and when it rains
    humidity = 80 - 1.3 * (air_temp - 24) + 0.8 * rain_mm + rng.normal(0, 4, hours)
    humidity = np.clip(humidity, 15, 100)

    # Soil moisture simulation (hour-by-hour water balance)
    soil = np.empty(hours)
    soil[0] = rng.uniform(35, 50)
    for t in range(1, hours):
        evaporation = 0.06 * max(air_temp[t] - 10, 0) * (1 + light_lux[t] / 60000) \
                      * (1 - humidity[t] / 150)
        soil[t] = soil[t - 1] - evaporation + 0.9 * rain_mm[t]
        # Farmer irrigates when the field gets dry (adds water), so the data looks like a real field
        if soil[t] < config.IRRIGATION_THRESHOLD and rng.random() < 0.25:
            soil[t] += rng.uniform(8, 12)
        soil[t] = np.clip(soil[t], 5, 65)

    # Sensor noise
    soil_measured = soil + rng.normal(0, 0.6, hours)

    # Simulated faults: 1% of readings are missing (kept as NaN, handled later)
    faults = rng.random(hours) < 0.01
    soil_measured[faults] = np.nan

    # Label: will the soil be below threshold in the next hour?
    next_soil = np.append(soil[1:], soil[-1])
    irrigation_needed = (next_soil < config.IRRIGATION_THRESHOLD).astype(int)

    return pd.DataFrame({
        "timestamp": timestamps,
        "device_id": device_id,
        "soil_moisture": np.round(soil_measured, 2),
        "air_temp": np.round(air_temp, 2),
        "humidity": np.round(humidity, 2),
        "light_lux": np.round(light_lux, 1),
        "rain_mm": np.round(rain_mm, 2),
        config.TARGET: irrigation_needed,
    })


def generate(hours: int = config.DATASET_HOURS, devices: int = config.NUM_DEVICES) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    frames = [simulate_device(f"FIELD-{i + 1:02d}", hours, rng) for i in range(devices)]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv(config.DATASET_PATH, index=False)
    print(f"Saved {len(df):,} rows to {config.DATASET_PATH}")
    print(f"Irrigation needed in {df[config.TARGET].mean() * 100:.1f}% of readings")
    print(df.head())
