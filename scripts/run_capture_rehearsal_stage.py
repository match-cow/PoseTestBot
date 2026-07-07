#!/usr/bin/env python3
"""Run a fake-iiwa pose-only capture rehearsal from run_config.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pipeline.capture_rehearsal import run_capture_rehearsal
from posetestbot.pipeline.run_config import load_run_config, load_run_config_for_run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start fake_iiwa_controller.py and pose_receiver_udp_json.py to "
            "exercise robot pose capture without starting camera hardware."
        )
    )
    parser.add_argument("run_root", help="Run folder containing run_config.json.")
    parser.add_argument(
        "--run-config",
        default=None,
        help="Optional run_config.json path. Defaults to <run_root>/run_config.json.",
    )
    parser.add_argument("--duration", type=float, default=0.3)
    parser.add_argument("--sample-ms", type=float, default=25.0)
    parser.add_argument("--startup-delay", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--robot-port", type=int, default=None)
    parser.add_argument("--receiver-port", type=int, default=None)
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--receiver-ip", default=None)
    parser.add_argument(
        "--controller-startup-wait",
        type=float,
        default=0.2,
        help="Seconds to wait for the fake controller socket before starting the receiver.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full rehearsal report JSON after writing it.",
    )
    return parser.parse_args()


def load_config(run_root: Path, run_config: str | None) -> dict:
    if run_config is None:
        return load_run_config_for_run_root(run_root)

    config = load_run_config(run_config)
    config_run_root = Path(str(config["run_root"])).resolve()
    if config_run_root != run_root.resolve():
        raise ValueError(
            "Run config run_root does not match requested run_root: "
            f"{config['run_root']} != {run_root.as_posix()}"
        )
    return config


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    config = load_config(run_root, args.run_config)
    path, report = run_capture_rehearsal(
        config,
        duration_s=args.duration,
        sample_ms=args.sample_ms,
        startup_delay_s=args.startup_delay,
        timeout_s=args.timeout_s,
        robot_port=args.robot_port,
        receiver_port=args.receiver_port,
        robot_ip=args.robot_ip,
        receiver_ip=args.receiver_ip,
        controller_startup_wait_s=args.controller_startup_wait,
    )

    print(f"Wrote {path}")
    print(f"Captured {report['raw_pose_count']} fake robot pose packet(s)")
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
