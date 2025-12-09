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


def _default_drones() -> List[str]:
    env_value = os.getenv("DRONE_IDS")
    if env_value:
        return [item.strip() for item in env_value.split(",") if item.strip()]
    return ["drone-A", "drone-B"]


def env_or_default(name: str, default: str) -> str:
    return os.getenv(name, default)


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
        default=_default_drones(),
        help="Drone IDs to simulate.",
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
                        yield _normalize_feature_row(row)
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
                        yield _normalize_feature_row(record)
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
                features = next(dataset_iter)
            except StopIteration:
                logger.info("[%s] Dataset finished, stopping producer loop.", drone_id)
                break
        else:
            features = generate_random_flow()

        payload = {
            "drone_id": drone_id,
            "timestamp": time.time(),
            "features": features,
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
    drone_ids: Iterable[str],
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
) -> None:
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
        )
    logger.info(
        "Started drone producer bootstrap=%s topic=%s drones=%s",
        bootstrap_servers,
        topic,
        list(drone_ids),
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
            for idx, drone_id in enumerate(drone_ids)
        ]
        await asyncio.gather(*tasks)
    finally:
        await producer.stop()
        logger.info("Drone producer stopped.")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        asyncio.run(
            run_simulation(
                drone_ids=args.drones,
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
            )
        )
    except KeyboardInterrupt:
        logger.info("Drone simulator interrupted by user.")


if __name__ == "__main__":
    main()
