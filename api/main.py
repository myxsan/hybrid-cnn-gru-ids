# api/main.py
import asyncio
import contextlib
import json
import logging
import os
import time
from collections import deque
from typing import Deque, Dict, List, Optional

from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

from model.ids_model import IDSService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tower")


class TowerConfig:
    """
    Minimal runtime configuration for the tower service.
    Values can be overridden via environment variables.
    """

    def __init__(_self):
        _self.kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        _self.kafka_flow_topic = os.getenv("KAFKA_FLOW_TOPIC", "flows")
        _self.kafka_group_id = os.getenv("KAFKA_GROUP_ID", "ids-tower")
        _self.consumer_retry_backoff = float(os.getenv("KAFKA_CONSUMER_RETRY_SEC", "5"))
        _self.event_history_size = int(os.getenv("TOWER_EVENT_HISTORY", "1000"))


CONFIG = TowerConfig()

LABEL_MAP = {0: "normal", 1: "attack"}

app = FastAPI(
    title="Drone IDS Tower",
    description="Central server that evaluates drone network traffic using CNN-GRU IDS.",
    version="1.2.0",
)

# Load model + scaler once at startup
ids_service = IDSService(device="cpu")

# In-memory event buffer consumed by /events and /dashboard.
events: Deque[Dict] = deque(maxlen=CONFIG.event_history_size)

# === Metrics ===
FLOW_COUNTER = Counter(
    "ids_flow_events_total",
    "Total inference events processed by the tower",
    ["source", "label"],
)
INFERENCE_LATENCY = Histogram(
    "ids_inference_latency_seconds",
    "Inference latency per source",
    ["source"],
)
EVENT_BUFFER_SIZE = Gauge(
    "ids_event_buffer_size",
    "Current number of events stored in the in-memory buffer",
)
KAFKA_CONSUMER_STATUS = Gauge(
    "ids_kafka_consumer_status",
    "Kafka consumer running indicator (1=running,0=stopped)",
)


DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Drone IDS Tower Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            color-scheme: dark;
            --bg: #0d1117;
            --panel: #151b23;
            --border: #30363d;
            --text: #c9d1d9;
            --muted: #8b949e;
            --accent: #58a6ff;
            --attack: #f85149;
            --benign: #3fb950;
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 24px;
            font-family: 'Space Grotesk', Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
        }}
        header {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }}
        h1 {{
            margin: 0;
            font-size: 1.8rem;
        }}
        .status-pill {{
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            background: var(--panel);
            border: 1px solid var(--border);
            color: var(--accent);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin: 24px 0;
        }}
        .card {{
            padding: 16px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
        }}
        .card span {{
            display: block;
            font-size: 0.85rem;
            color: var(--muted);
        }}
        .card strong {{
            display: block;
            font-size: 1.8rem;
            margin-top: 6px;
        }}
        .table-wrapper {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            overflow: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
        }}
        th, td {{
            padding: 10px;
            border-bottom: 1px solid var(--border);
            text-align: left;
        }}
        th {{
            color: var(--muted);
            font-weight: 500;
        }}
        tr.attack {{
            background: rgba(248, 81, 73, 0.08);
        }}
        tr.normal {{
            background: rgba(63, 185, 80, 0.05);
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .legend {{
            display: flex;
            gap: 12px;
            margin: 8px 0 16px;
            font-size: 0.85rem;
            color: var(--muted);
        }}
        .legend span {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        .dot.attack {{
            background: var(--attack);
        }}
        .dot.normal {{
            background: var(--benign);
        }}
        @media (max-width: 600px) {{
            header {{
                flex-direction: column;
                align-items: flex-start;
            }}
            table {{
                font-size: 0.85rem;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Drone IDS Tower</h1>
            <p class="muted">Kafka topic: {topic} &mdash; Bootstrap: {bootstrap}</p>
        </div>
        <div class="status-pill" id="connection-status">Starting…</div>
    </header>

    <div class="grid">
        <div class="card">
            <span>Total Events</span>
            <strong id="stat-total">0</strong>
        </div>
        <div class="card">
            <span>Attack %</span>
            <strong id="stat-attack">0%</strong>
        </div>
        <div class="card">
            <span>Avg Latency</span>
            <strong id="stat-latency">0 ms</strong>
        </div>
        <div class="card">
            <span>Last Update</span>
            <strong id="stat-updated">–</strong>
        </div>
    </div>

    <div class="table-wrapper">
        <div class="legend">
            <span><span class="dot normal"></span>Normal</span>
            <span><span class="dot attack"></span>Attack</span>
            <span id="summary-active"></span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Drone</th>
                    <th>Source</th>
                    <th>Flow Timestamp</th>
                    <th>Tower Timestamp</th>
                    <th>Label</th>
                    <th>Probabilities</th>
                    <th>Latency</th>
                </tr>
            </thead>
            <tbody id="events-body">
                <tr><td colspan="7">Waiting for events…</td></tr>
            </tbody>
        </table>
    </div>

    <script>
    const REFRESH_INTERVAL = 4000;
    const eventsBody = document.getElementById("events-body");
    const statusEl = document.getElementById("connection-status");
    const statTotal = document.getElementById("stat-total");
    const statAttack = document.getElementById("stat-attack");
    const statLatency = document.getElementById("stat-latency");
    const statUpdated = document.getElementById("stat-updated");
    const summaryActive = document.getElementById("summary-active");

    function formatProbabilities(probabilities) {{
        return probabilities
            .map((p, idx) => `P${{idx}}=${{(p * 100).toFixed(1)}}%`)
            .join(" / ");
    }}

    function formatTimestamp(ts) {{
        if (!ts) {{
            return "–";
        }}
        const date = new Date(ts * 1000);
        return date.toLocaleString();
    }}

    function updateStats(events) {{
        const total = events.length;
        const attackCount = events.filter(ev => ev.prediction.class_label === "attack").length;
        const avgLatency = total
            ? (events.reduce((acc, ev) => acc + ev.inference_latency_sec, 0) / total) * 1000
            : 0;
        const drones = new Set(events.map(ev => ev.drone_id));

        statTotal.textContent = total.toString();
        statAttack.textContent = total ? `${{((attackCount / total) * 100).toFixed(1)}}%` : "0%";
        statLatency.textContent = `${{avgLatency.toFixed(1)}} ms`;
        statUpdated.textContent = new Date().toLocaleTimeString();
        summaryActive.textContent = `${{drones.size}} active drones`;
    }}

    function renderEvents(events) {{
        if (!events.length) {{
            eventsBody.innerHTML = "<tr><td colspan='7'>No events in window.</td></tr>";
            return;
        }}
        eventsBody.innerHTML = events
            .map(ev => {{
                const label = ev.prediction.class_label;
                return `
                <tr class="${{label}}">
                    <td>${{ev.drone_id}}</td>
                    <td>${{ev.source}}</td>
                    <td>${{formatTimestamp(ev.flow_timestamp)}}</td>
                    <td>${{formatTimestamp(ev.tower_timestamp)}}</td>
                    <td>${{label}}</td>
                    <td>${{formatProbabilities(ev.prediction.probabilities)}}</td>
                    <td>${{(ev.inference_latency_sec * 1000).toFixed(2)}} ms</td>
                </tr>`;
            }})
            .join("");
    }}

    async function refresh() {{
        try {{
            const response = await fetch("/events?limit=200");
            if (!response.ok) throw new Error("Request failed");
            const payload = await response.json();
            const events = payload.events || [];
            statusEl.textContent = "Streaming";
            statusEl.style.color = "var(--accent)";
            updateStats(events);
            renderEvents(events);
        }} catch (err) {{
            statusEl.textContent = "Disconnected";
            statusEl.style.color = "var(--attack)";
            console.error(err);
        }}
    }}

    refresh();
    setInterval(refresh, REFRESH_INTERVAL);
    </script>
</body>
</html>
"""


# === Request / Response models ===
class FlowFeatures(BaseModel):
    drone_id: str
    timestamp: Optional[float] = None
    features: Dict[str, float]


class BatchFlowFeatures(BaseModel):
    flows: List[FlowFeatures]


def _record_event(
    *,
    drone_id: str,
    result: Dict,
    latency: float,
    source: str,
    client_timestamp: Optional[float],
) -> Dict:
    """
    Normalize prediction output, store it in the events deque, and return
    a serializable representation for API consumers.
    """
    pred_class = result["pred_class"]
    tower_timestamp = time.time()
    event = {
        "drone_id": drone_id,
        "source": source,
        "flow_timestamp": client_timestamp,
        "tower_timestamp": tower_timestamp,
        "inference_latency_sec": latency,
        "prediction": {
            "class_id": pred_class,
            "class_label": LABEL_MAP.get(pred_class, "unknown"),
            "probabilities": result["probabilities"],
        },
    }
    events.append(event)
    EVENT_BUFFER_SIZE.set(len(events))
    FLOW_COUNTER.labels(source=source, label=event["prediction"]["class_label"]).inc()
    INFERENCE_LATENCY.labels(source=source).observe(latency)
    return event


async def process_flow_message(payload: Dict, source: str = "kafka") -> None:
    """
    Shared logic for Kafka-consumed flows.
    Runs inference inside a worker thread to avoid blocking the event loop.
    """
    if not isinstance(payload, dict):
        logger.warning("Skipping malformed payload (not a dict): %s", payload)
        return

    features = payload.get("features")
    drone_id = payload.get("drone_id", "unknown-drone")
    client_ts = payload.get("timestamp")

    if not isinstance(features, dict):
        logger.warning("Skipping payload with missing features for drone %s", drone_id)
        return

    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(ids_service.predict_single, features)
    except Exception:
        logger.exception("Failed to run inference for drone %s", drone_id)
        return

    latency = time.perf_counter() - start
    event = _record_event(
        drone_id=drone_id,
        result=result,
        latency=latency,
        source=source,
        client_timestamp=client_ts,
    )
    logger.info(
        "Flow processed source=%s drone=%s label=%s latency=%.4fs",
        source,
        drone_id,
        event["prediction"]["class_label"],
        latency,
    )


async def kafka_consumer_worker():
    """
    Continuously consume flow messages from Kafka and feed them to the model.
    Retries with exponential-like backoff when brokers are unreachable.
    """
    backoff = CONFIG.consumer_retry_backoff
    while True:
        consumer = AIOKafkaConsumer(
            CONFIG.kafka_flow_topic,
            bootstrap_servers=CONFIG.kafka_bootstrap_servers,
            group_id=CONFIG.kafka_group_id,
            enable_auto_commit=True,
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        )
        try:
            await consumer.start()
            KAFKA_CONSUMER_STATUS.set(1)
            logger.info(
                "Kafka consumer listening on %s topic=%s",
                CONFIG.kafka_bootstrap_servers,
                CONFIG.kafka_flow_topic,
            )
            async for msg in consumer:
                await process_flow_message(msg.value, source="kafka")
        except asyncio.CancelledError:
            logger.info("Kafka consumer task cancelled. Shutting down.")
            raise
        except Exception:
            logger.exception("Kafka consumer error. Retrying in %.1fs", backoff)
            await asyncio.sleep(backoff)
        finally:
            with contextlib.suppress(Exception):
                await consumer.stop()
            KAFKA_CONSUMER_STATUS.set(0)


@app.on_event("startup")
async def startup_event():
    app.state.kafka_task = asyncio.create_task(kafka_consumer_worker())


@app.on_event("shutdown")
async def shutdown_event():
    task = getattr(app.state, "kafka_task", None)
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Drone IDS Tower is running.",
        "kafka": {
            "bootstrap_servers": CONFIG.kafka_bootstrap_servers,
            "topic": CONFIG.kafka_flow_topic,
            "group_id": CONFIG.kafka_group_id,
        },
        "total_events": len(events),
    }


@app.get("/healthz")
def healthz():
    kafka_task = getattr(app.state, "kafka_task", None)
    kafka_running = bool(kafka_task and not kafka_task.done())
    return {
        "status": "ok",
        "kafka_consumer_running": kafka_running,
        "event_buffer_size": len(events),
    }


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(flow: FlowFeatures):
    """
    Single flow prediction via HTTP.
    Also records the result in the shared events buffer.
    """
    start = time.perf_counter()
    try:
        result = ids_service.predict_single(flow.features)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing feature: {exc}") from exc

    latency = time.perf_counter() - start
    event = _record_event(
        drone_id=flow.drone_id,
        result=result,
        latency=latency,
        source="http",
        client_timestamp=flow.timestamp,
    )
    return event


@app.post("/predict_batch")
def predict_batch(batch: BatchFlowFeatures):
    """
    Predict many flows in one request.
    """
    responses = []
    for flow in batch.flows:
        start = time.perf_counter()
        try:
            result = ids_service.predict_single(flow.features)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing feature: {exc}") from exc
        latency = time.perf_counter() - start
        responses.append(
            _record_event(
                drone_id=flow.drone_id,
                result=result,
                latency=latency,
                source="http-batch",
                client_timestamp=flow.timestamp,
            )
        )

    return {"results": responses}


@app.get("/events")
def get_events(limit: int = 100):
    """
    Return the most recent inference events for dashboard or API consumers.
    """
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be positive")
    limit = min(limit, CONFIG.event_history_size)
    recent = list(events)[-limit:]
    return {"events": list(reversed(recent))}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html = DASHBOARD_TEMPLATE.format(
        topic=CONFIG.kafka_flow_topic,
        bootstrap=CONFIG.kafka_bootstrap_servers,
    )
    return HTMLResponse(content=html)
