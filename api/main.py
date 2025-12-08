# api/main.py
from typing import Dict, List, Optional
import time

from fastapi import FastAPI
from pydantic import BaseModel

from model.ids_model import IDSService


app = FastAPI(
    title="Drone IDS Tower",
    description="Central server that evaluates drone network traffic using CNN-GRU IDS.",
    version="1.0.0",
)

# Load model + scaler once at startup
ids_service = IDSService(device="cpu")


# === Request / Response models ===
class FlowFeatures(BaseModel):
    drone_id: str                       
    timestamp: Optional[float] = None  
    features: Dict[str, float]         


class BatchFlowFeatures(BaseModel):
    flows: List[FlowFeatures]


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Drone IDS Tower is running.",
    }


@app.post("/predict")
def predict(flow: FlowFeatures):
    """
    Single flow prediction.
    One drone sends one processed flow to the tower.
    """
    start = time.time()
    result = ids_service.predict_single(flow.features)
    latency = time.time() - start

    # simple mapping: 0 = normal, 1 = attack (adjust if needed)
    label_map = {0: "normal", 1: "attack"}

    pred_class = result["pred_class"]
    response = {
        "drone_id": flow.drone_id,
        "tower_timestamp": time.time(),
        "inference_latency_sec": latency,
        "prediction": {
            "class_id": pred_class,
            "class_label": label_map.get(pred_class, "unknown"),
            "probabilities": result["probabilities"],
        },
    }
    return response


@app.post("/predict_batch")
def predict_batch(batch: BatchFlowFeatures):
    """
    Optional: many flows from many drones in one request.
    For now we simply loop predict_single.
    """
    responses = []
    for flow in batch.flows:
        start = time.time()
        result = ids_service.predict_single(flow.features)
        latency = time.time() - start

        label_map = {0: "normal", 1: "attack"}
        pred_class = result["pred_class"]

        responses.append({
            "drone_id": flow.drone_id,
            "tower_timestamp": time.time(),
            "inference_latency_sec": latency,
            "prediction": {
                "class_id": pred_class,
                "class_label": label_map.get(pred_class, "unknown"),
                "probabilities": result["probabilities"],
            },
        })

    return {"results": responses}
