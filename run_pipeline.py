"""
Step 5 - Run the whole real-time pipeline:
    simulated IoT sensors  --queue-->  edge node (ML prediction)  -->  results/predictions.csv

Usage:
    python run_pipeline.py                     # simulate 300 readings, 0.5 s apart
    python run_pipeline.py --rows 1000 --interval 0.1
    python run_pipeline.py --rows 50000 --interval 0   # fastest, for a full batch

Open the dashboard in a second terminal with:  streamlit run dashboard.py
"""
import argparse
import queue

import config
import generate_dataset
import train_model
from edge_node import EdgeNode
from iot_simulator import SensorSimulator
import model_registry


def ensure_ready():
    """Create the dataset and model automatically if this is the first run."""
    if not config.DATASET_PATH.exists():
        print("Dataset missing - generating it now...")
        generate_dataset.generate().to_csv(config.DATASET_PATH, index=False)
    if not model_registry.list_versions():
        print("No model registered yet - training now...")
        train_model.train()


def main():
    parser = argparse.ArgumentParser(description="Smart Agriculture edge ML pipeline")
    parser.add_argument("--rows", type=int, default=300, help="number of sensor readings to stream")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="seconds between readings (0 = as fast as possible)")
    parser.add_argument("--fault-rate", type=float, default=0.02,
                        help="fraction of readings that are faulty (for validation demo)")
    args = parser.parse_args()

    ensure_ready()

    sensor_queue: queue.Queue = queue.Queue(maxsize=1000)
    edge = EdgeNode(sensor_queue)
    simulator = SensorSimulator(sensor_queue, interval_sec=args.interval,
                                max_rows=args.rows, fault_rate=args.fault_rate)

    print(f"Streaming {args.rows} readings from {config.NUM_DEVICES} devices "
          f"({args.interval}s interval)...")
    simulator.start()
    try:
        edge.run()  # blocks until the simulator sends the end-of-stream marker
    except KeyboardInterrupt:
        print("\nStopping...")
        simulator.stop()

    print(f"Results written to {config.PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
