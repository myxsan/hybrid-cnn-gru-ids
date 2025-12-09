# nodes/drone_producer.py
import argparse
import asyncio
import contextlib
import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drone-producer")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_FEATURES = PROJECT_ROOT / "artifacts" / "feature_order.json"

with ARTIFACT_FEATURES.open() as f:
    FEATURE_NAMES: Sequence[str] = json.load(f)


def env_or_default(name: str, default: str) -> str:
    return os.getenv(name, default)


def generate_drone_ids(count: int) -> List[str]:
    count = max(2, min(10, count))
    return [f"DRONE-{idx}" for idx in range(1, count + 1)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kafka drone flow producer")
    parser.add_argument(
        "--bootstrap-servers",
        default=env_or_default("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap servers (comma separated).",
    )
    parser.add_argument(
        "--topic",
        default=env_or_default("KAFKA_FLOW_TOPIC", "flows"),
        help="Kafka topic for flow messages.",
    )
    parser.add_argument(
        "--drones",
        nargs="+",
        help="Drone IDs to simulate (overrides --drone-count).",
    )
    parser.add_argument(
        "--drone-count",
        type=int,
        default=int(env_or_default("DRONE_COUNT", "3")),
        help="Number of drones to auto-generate (2-10).",
    )
    parser.add_argument(
        "--flows-per-drone",
        type=int,
        default=int(env_or_default("FLOWS_PER_DRONE", "0")),
        help="Number of flows per drone (0 = infinite).",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=float(env_or_default("FLOW_INTERVAL_SEC", "1.0")),
        help="Average delay between two flows from the same drone.",
    )
    parser.add_argument(
        "--jitter-sec",
        type=float,
        default=float(env_or_default("FLOW_JITTER_SEC", "0.2")),
        help="Random jitter (+/-) added to the interval.",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=float(env_or_default("DRONE_STARTUP_DELAY", "0.0")),
        help="Optional delay before each drone starts sending.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=env_or_default("DRONE_DATASET_PATH", ""),
        help="Optional path to a CSV or JSONL file with recorded flows.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["csv", "jsonl"],
        default=env_or_default("DRONE_DATASET_FORMAT", "csv"),
        help="Format of the dataset file when --dataset-path is provided.",
    )
    parser.add_argument(
        "--dataset-repeat",
        action="store_true",
        default=env_or_default("DRONE_DATASET_REPEAT", "false").lower() == "true",
        help="Replay the dataset in a loop instead of stopping at EOF.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=env_or_default("DRONE_LABEL_COLUMN", "Label"),
        help="Column name carrying the ground-truth label (optional).",
    )
    parser.add_argument(
        "--connect-retry-sec",
        type=float,
        default=float(env_or_default("KAFKA_CONNECT_RETRY_SEC", "5.0")),
        help="Initial backoff when Kafka is unavailable.",
    )
    parser.add_argument(
        "--connect-retry-max-sec",
        type=float,
        default=float(env_or_default("KAFKA_CONNECT_RETRY_MAX_SEC", "30.0")),
        help="Maximum backoff when retrying Kafka connections.",
    )
    return parser


def generate_random_flow() -> dict:
    """
    Create a random feature vector compatible with the IDS model.
    Values are sampled uniformly in [0,1] per feature.
    """
    return {name: random.random() for name in FEATURE_NAMES}


def normalize_label(raw_value: Optional[str]) -> Optional[str]:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    lower = value.lower()
    normal_tokens = {"benign", "normal", "benign traffic", "0"}
    attack_tokens = {"1", "attack", "malicious", "anomaly"}
    if lower in normal_tokens:
        return "normal"
    if lower in attack_tokens:
        return "attack"
    try:
        return "attack" if float(value) >= 0.5 else "normal"
    except ValueError:
        return "attack"


def _normalize_feature_row(row: dict) -> dict:
    """
    Ensure the row contains every feature and convert values to float.
    """
    features = {}
    source = row.get("features", row)
    for name in FEATURE_NAMES:
        try:
            features[name] = float(source[name])
        except KeyError as exc:
            raise ValueError(f"missing feature {name}") from exc
        except ValueError as exc:
            raise ValueError(f"invalid float for {name}") from exc
    return features


def dataset_flow_generator(
    *,
    path: Path,
    fmt: str,
    repeat: bool,
    label_column: Optional[str],
) -> Iterator[dict]:
    """
    Yield flows from a dataset file. Supports CSV (wide format) or JSON Lines.
    """
    while True:
        if fmt == "csv":
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        label = None
                        if label_column and label_column in row:
                            label = normalize_label(row[label_column])
                        yield {
                            "features": _normalize_feature_row(row),
                            "true_label": label,
                        }
                    except ValueError as exc:
                        logger.warning("Skipping CSV row: %s", exc)
        else:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        label = None
                        if label_column and isinstance(record, dict):
                            label = normalize_label(record.get(label_column))
                        yield {
                            "features": _normalize_feature_row(record),
                            "true_label": label,
                        }
                    except (json.JSONDecodeError, ValueError) as exc:
                        logger.warning("Skipping JSON line: %s", exc)
        if not repeat:
            break


async def _drone_loop(
    *,
    producer: AIOKafkaProducer,
    drone_id: str,
    topic: str,
    flows: int,
    interval: float,
    jitter: float,
    startup_delay: float = 0.0,
    dataset_iterator_factory: Optional[Callable[[], Iterator[dict]]] = None,
) -> None:
    if startup_delay > 0:
        await asyncio.sleep(startup_delay)

    dataset_iter = dataset_iterator_factory() if dataset_iterator_factory else None
    sent = 0
    while flows <= 0 or sent < flows:
        if dataset_iter is not None:
            try:
                sample = next(dataset_iter)
            except StopIteration:
                logger.info("[%s] Dataset finished, stopping producer loop.", drone_id)
                break
            features = sample["features"]
            true_label = sample.get("true_label")
        else:
            features = generate_random_flow()
            true_label = None

        payload = {
            "drone_id": drone_id,
            "timestamp": time.time(),
            "features": features,
            "true_label": true_label,
        }
        try:
            await producer.send_and_wait(topic, payload)
            logger.info(
                "[%s] flow sent to topic=%s remaining=%s",
                drone_id,
                topic,
                "inf" if flows <= 0 else flows - sent - 1,
            )
        except Exception:
            logger.exception("Failed to send flow for drone %s", drone_id)
            await asyncio.sleep(interval)
            continue

        sent += 1
        delay = interval
        if jitter > 0:
            delay += random.uniform(-jitter, jitter)
            delay = max(0.01, delay)
        await asyncio.sleep(delay)


async def _start_producer_with_retry(
    *,
    bootstrap_servers: str,
    serializer,
    initial_backoff: float,
    max_backoff: float,
) -> AIOKafkaProducer:
    backoff = max(0.5, initial_backoff)
    attempt = 1
    while True:
        producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=serializer,
            linger_ms=10,
        )
        try:
            await producer.start()
            return producer
        except KafkaConnectionError as exc:
            logger.warning(
                "Kafka unreachable (%s). attempt=%d retry_in=%.1fs",
                exc,
                attempt,
                backoff,
            )
        except Exception:
            logger.exception("Unexpected error starting Kafka producer.")
        with contextlib.suppress(Exception):
            await producer.stop()
        await asyncio.sleep(backoff)
        attempt += 1
        backoff = min(backoff * 2, max_backoff)


async def run_simulation(
    *,
    drone_ids: Optional[Iterable[str]],
    bootstrap_servers: str,
    topic: str = "flows",
    flows_per_drone: int = 0,
    interval_sec: float = 1.0,
    jitter_sec: float = 0.2,
    startup_delay: float = 0.0,
    connect_retry_sec: float = 5.0,
    connect_retry_max_sec: float = 30.0,
    dataset_path: Optional[str] = None,
    dataset_format: str = "csv",
    dataset_repeat: bool = False,
    label_column: Optional[str] = None,
    drone_count: int = 3,
) -> None:
    selected_drones = list(drone_ids) if drone_ids else generate_drone_ids(drone_count)

    producer = await _start_producer_with_retry(
        bootstrap_servers=bootstrap_servers,
        serializer=lambda value: json.dumps(value).encode("utf-8"),
        initial_backoff=connect_retry_sec,
        max_backoff=connect_retry_max_sec,
    )
    dataset_factory: Optional[Callable[[], Iterator[dict]]] = None
    if dataset_path:
        resolved = Path(dataset_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset path not found: {resolved}")
        logger.info(
            "Using dataset for flows path=%s format=%s repeat=%s",
            resolved,
            dataset_format,
            dataset_repeat,
        )
        dataset_factory = lambda: dataset_flow_generator(
            path=resolved,
            fmt=dataset_format,
            repeat=dataset_repeat,
            label_column=label_column,
        )
    logger.info(
        "Started drone producer bootstrap=%s topic=%s drones=%s",
        bootstrap_servers,
        topic,
        selected_drones,
    )
    try:
        tasks = [
            asyncio.create_task(
                _drone_loop(
                    producer=producer,
                    drone_id=drone_id,
                    topic=topic,
                    flows=flows_per_drone,
                    interval=interval_sec,
                    jitter=jitter_sec,
                    startup_delay=startup_delay * idx,
                    dataset_iterator_factory=dataset_factory,
                )
            )
            for idx, drone_id in enumerate(selected_drones)
        ]
        await asyncio.gather(*tasks)
    finally:
        await producer.stop()
        logger.info("Drone producer stopped.")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.drones:
        drone_ids = args.drones
    else:
        drone_ids = generate_drone_ids(args.drone_count)
    try:
        asyncio.run(
            run_simulation(
                drone_ids=drone_ids,
                bootstrap_servers=args.bootstrap_servers,
                topic=args.topic,
                flows_per_drone=args.flows_per_drone,
                interval_sec=args.interval_sec,
                jitter_sec=args.jitter_sec,
                startup_delay=args.startup_delay,
                connect_retry_sec=args.connect_retry_sec,
                connect_retry_max_sec=args.connect_retry_max_sec,
                dataset_path=args.dataset_path or None,
                dataset_format=args.dataset_format,
                dataset_repeat=args.dataset_repeat,
                label_column=args.label_column or None,
                drone_count=args.drone_count,
            )
        )
    except KeyboardInterrupt:
        logger.info("Drone simulator interrupted by user.")


if __name__ == "__main__":
    main()
