# simulation/simulate_drones.py
import json
import random
import time
from pathlib import Path

import requests


TOWER_URL = "http://localhost:8000/predict"

# Load feature names so we know what keys to send
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT_ROOT / "artifacts" / "feature_order.json"

with FEATURE_FILE.open() as f:
    FEATURE_NAMES = json.load(f)


def generate_random_flow() -> dict:
    """
    For now: generate random values in [0, 1] for each feature.
    Later you can plug real CIC flows here.
    """
    return {name: random.random() for name in FEATURE_NAMES}


def send_flow(drone_id: str, flow_features: dict):
    payload = {
        "drone_id": drone_id,
        "timestamp": time.time(),
        "features": flow_features,
    }

    try:
        resp = requests.post(TOWER_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"[{drone_id}] ERROR sending flow: {e}")
        return

    if resp.status_code != 200:
        print(f"[{drone_id}] ERROR {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    pred = data["prediction"]["class_label"]
    latency = data["inference_latency_sec"]
    print(
        f"[{drone_id}] -> prediction={pred}, "
        f"latency={latency:.4f}s, "
        f"probs={data['prediction']['probabilities']}"
    )


def simulate_drone(drone_id: str, n_flows: int, delay: float):
    print(f"--- Starting simulation for {drone_id} ---")
    for i in range(n_flows):
        flow = generate_random_flow()
        send_flow(drone_id, flow)
        time.sleep(delay)
    print(f"--- {drone_id} finished ---\n")


if __name__ == "__main__":
    # simple sequential simulation, you can later use threading for parallel drones
    simulate_drone("drone-A", n_flows=5, delay=0.5)
    simulate_drone("drone-B", n_flows=5, delay=0.3)
    simulate_drone("drone-C", n_flows=5, delay=0.2)
