# simulation/simulate_drones.py
"""
Local helper to push a few sample flows into Kafka without Docker.
"""
import asyncio
import os

from nodes.drone_producer import run_simulation


if __name__ == "__main__":
    asyncio.run(
        run_simulation(
            drone_ids=["drone-A", "drone-B", "drone-C"],
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic=os.getenv("KAFKA_FLOW_TOPIC", "flows"),
            flows_per_drone=5,
            interval_sec=0.4,
            jitter_sec=0.1,
        )
    )
