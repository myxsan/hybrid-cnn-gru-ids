# hybrid-cnn-gru-ids

Distributed IDS tower powered by a hybrid CNN–GRU PyTorch model. Drones stream
flow features through Kafka, the tower consumes those flows asynchronously, and
a FastAPI dashboard exposes predictions in real time.

## Repository layout

- `api/` – FastAPI “tower” service with Kafka consumer + dashboard.
- `model/` – IDSService, scaler loading, and CNN–GRU architecture.
- `nodes/` – Kafka drone producer utilities and Dockerfile.
- `simulation/` – Local helper scripts for ad-hoc testing (ignored in Docker builds).
- `artifacts/` – Saved PyTorch weights, scaler, and feature ordering.

## Running everything with Docker Compose

1. Build and start the stack:
   ```bash
   docker compose up --build
   ```
   This launches:
   - Zookeeper + Kafka (Bitnami images, linux/amd64)
   - `ids-tower` (FastAPI + aiokafka consumer)
   - `drone-simulator` (optional random-flow producer)

2. Visit the tower:
   - API: http://localhost:8000/docs
   - Dashboard: http://localhost:8000/dashboard

3. Stop the stack with `CTRL+C` or `docker compose down`.

Environment variables (most already set in `docker-compose.yml`):

| Service | Variable | Description |
| --- | --- | --- |
| ids-tower | `KAFKA_BOOTSTRAP_SERVERS` | Kafka bootstrap servers (`kafka:9092`) |
| ids-tower | `KAFKA_FLOW_TOPIC` | Kafka topic to consume flows (`flows`) |
| ids-tower | `KAFKA_GROUP_ID` | Consumer group id (`ids-tower`) |
| ids-tower | `TOWER_EVENT_HISTORY` | Max in-memory events kept for dashboard |
| drone-simulator | `DRONE_IDS` | Comma-separated drone IDs |
| drone-simulator | `FLOWS_PER_DRONE` | Number of flows per drone (`0` = infinite) |
| drone-simulator | `FLOW_INTERVAL_SEC` | Average delay between flows |
| drone-simulator | `FLOW_JITTER_SEC` | +/- jitter applied to intervals |
| drone-simulator | `DRONE_DATASET_PATH` | Optional CSV/JSONL file for replaying real flows |
| drone-simulator | `DRONE_DATASET_FORMAT` | `csv` or `jsonl` (default `csv`) |
| drone-simulator | `DRONE_DATASET_REPEAT` | Loop dataset when `true` |

## Running the drone producer locally

The drone simulator can be executed outside Docker once Kafka is up:

```bash
python -m nodes.drone_producer \
  --bootstrap-servers localhost:9092 \
  --topic flows \
  --drones drone-A drone-B drone-C \
  --flows-per-drone 10 \
  --interval-sec 0.5 \
  --jitter-sec 0.1
```

The script reads `feature_order.json` to build valid payloads automatically. To replay a portion of the CIC-IDS2017 dataset (or any CSV/JSONL matching the feature order), pass the dataset path:

```bash
python -m nodes.drone_producer \
  --bootstrap-servers localhost:9092 \
  --topic flows \
  --dataset-path path/to/cic_sample.csv \
  --dataset-format csv \
  --dataset-repeat \
  --flows-per-drone 0
```

The CSV must contain one column per feature (matching `artifacts/feature_order.json`). JSON Lines can either include a `features` object or present the features as the top-level keys.

For a quick “burst” of flows during development, run:

```bash
python simulation/simulate_drones.py
```

That helper simply calls the producer with a short run (5 flows per drone).

## Observability

- The dashboard at http://localhost:8000/dashboard now delivers live updates via JavaScript with richer summaries (active drones, attack rate, latency).
- `/events` still returns the latest inference events for custom tooling.
- `/healthz` exposes a lightweight readiness response (Kafka consumer status and buffer depth).
- `/metrics` exports Prometheus-compatible metrics:
  - `ids_flow_events_total{source,label}`
  - `ids_inference_latency_seconds_bucket/sum/count`
  - `ids_event_buffer_size`
  - `ids_kafka_consumer_status`

Scrape `http://ids-tower:8000/metrics` from your monitoring stack to track inference load and latency.
