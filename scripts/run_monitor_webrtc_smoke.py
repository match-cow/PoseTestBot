#!/usr/bin/env python3
"""Plan or run the safety-gated UGREEN room-monitor hardware smoke test."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from posetestbot.monitoring.smoke import (
    DEFAULT_EXPECTED_NODE,
    DEFAULT_FRAME_TARGET,
    DEFAULT_TIMEOUT_S,
    build_smoke_plan,
    run_monitor_webrtc_smoke,
)


APP_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run a monitor-only UGREEN WebRTC hardware smoke test."
    )
    parser.add_argument(
        "smoke_root",
        nargs="?",
        default=None,
        help="Output folder (defaults to a unique working_data folder).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the smoke plan without opening hardware or writing artifacts.",
    )
    parser.add_argument(
        "--operator-authorized",
        action="store_true",
        help="Confirm explicit operator authorization for this physical camera test.",
    )
    parser.add_argument(
        "--allow-cameras",
        action="store_true",
        help="Pass the camera execution safety gate.",
    )
    parser.add_argument(
        "--allow-real-robot",
        action="store_true",
        help="Pass the lab physical-execution gate; this command never contacts the robot.",
    )
    parser.add_argument("--expected-node", default=DEFAULT_EXPECTED_NODE)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_TARGET)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    return parser.parse_args()


def default_smoke_root() -> Path:
    return APP_ROOT / "working_data" / "monitor_webrtc_smoke" / uuid.uuid4().hex[:12]


def main() -> int:
    args = parse_args()
    smoke_root = Path(args.smoke_root) if args.smoke_root else default_smoke_root()
    plan = build_smoke_plan(
        smoke_root,
        expected_node=args.expected_node,
        frame_target=args.frames,
        timeout_s=args.timeout_s,
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    try:
        report_path, report = asyncio.run(
            run_monitor_webrtc_smoke(
                smoke_root,
                operator_authorized=args.operator_authorized,
                allow_cameras=args.allow_cameras,
                allow_real_robot=args.allow_real_robot,
                expected_node=args.expected_node,
                frame_target=args.frames,
                timeout_s=args.timeout_s,
                repo_root=APP_ROOT,
            )
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Monitor WebRTC smoke failed: {exc}")
        return 2
    print(f"Wrote {report_path}")
    print(
        "Monitor WebRTC smoke: "
        f"{report['status']} ({report['receiver']['received_frames']} frames)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
