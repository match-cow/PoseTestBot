#!/usr/bin/env python3
"""Write a manifest-tracked capture command plan for a configured run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posetestbot.pipeline.capture_plan import write_capture_plan_with_manifest
from posetestbot.pipeline.run_config import load_run_config, load_run_config_for_run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build capture_plan.json from run_config.json without starting cameras "
            "or robot motion."
        )
    )
    parser.add_argument("run_root", help="Run folder containing run_config.json.")
    parser.add_argument(
        "--run-config",
        default=None,
        help="Optional run_config.json path. Defaults to <run_root>/run_config.json.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max frame count to include in planned camera commands.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full capture plan JSON after writing it.",
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
    path, plan = write_capture_plan_with_manifest(
        run_root,
        config,
        run_config_path=args.run_config,
        max_frames=args.max_frames,
    )

    print(f"Wrote {path}")
    for command in sorted(plan.commands, key=lambda item: item.startup_order):
        print(f"[{command.startup_order}] {command.name}: {' '.join(command.command)}")
    if args.print_json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
