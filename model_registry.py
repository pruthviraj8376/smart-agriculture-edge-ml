"""
Simple local Model Registry.

Every trained model is saved as a versioned file (models/irrigation_predictor_v1.joblib, v2, ...)
and its metadata (metrics, features, training date, stage) is written to models/model_registry.json.
The edge node always loads the model whose stage is "Production".

Stages:
  Staging    - newly trained, under evaluation
  Production - the model the edge node uses
  Archived   - older versions kept for history
"""
import json
from datetime import datetime
from pathlib import Path

import joblib

import config


def _load_registry() -> dict:
    if config.REGISTRY_PATH.exists():
        return json.loads(config.REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"model_name": config.MODEL_NAME, "versions": []}


def _save_registry(registry: dict) -> None:
    config.REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def register_model(model, metrics: dict, features: list, params: dict, promote: bool = True) -> dict:
    """Save a trained model as a new version and (optionally) promote it to Production."""
    registry = _load_registry()
    version = len(registry["versions"]) + 1
    model_file = config.MODELS_DIR / f"{config.MODEL_NAME}_v{version}.joblib"
    joblib.dump(model, model_file)

    if promote:
        for v in registry["versions"]:
            if v["stage"] == "Production":
                v["stage"] = "Archived"

    entry = {
        "version": version,
        "file": model_file.name,
        "stage": "Production" if promote else "Staging",
        "registered_at": datetime.now().isoformat(timespec="seconds"),
        "features": features,
        "params": params,
        "metrics": metrics,
    }
    registry["versions"].append(entry)
    _save_registry(registry)
    return entry


def list_versions() -> list[dict]:
    return _load_registry()["versions"]


def get_production_model():
    """Return (model, metadata) for the version currently in Production."""
    registry = _load_registry()
    for entry in reversed(registry["versions"]):
        if entry["stage"] == "Production":
            model = joblib.load(config.MODELS_DIR / entry["file"])
            return model, entry
    raise RuntimeError("No Production model found. Run train_model.py first.")


def get_production_info() -> dict | None:
    for entry in reversed(list_versions()):
        if entry["stage"] == "Production":
            return entry
    return None


if __name__ == "__main__":
    for v in list_versions():
        print(f"v{v['version']} [{v['stage']}] registered {v['registered_at']} metrics={v['metrics']}")
