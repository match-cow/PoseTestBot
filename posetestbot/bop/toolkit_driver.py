"""Drive the pinned official BOP19 metric scripts for one custom dataset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from bop_toolkit_lib import dataset_params, inout, misc


VSD_AND_MSSD_GRID = tuple(float(value) for value in np.arange(0.05, 0.51, 0.05))
ERRORS = (
    {
        "type": "vsd",
        "thresholds": VSD_AND_MSSD_GRID,
        "taus": VSD_AND_MSSD_GRID,
    },
    {
        "type": "mssd",
        "thresholds": VSD_AND_MSSD_GRID,
        "taus": (),
    },
    {
        "type": "mspd",
        "thresholds": tuple(range(5, 51, 5)),
        "taus": (),
    },
)


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _run_score(command: list[str]) -> None:
    """Run one chatty score calculation while retaining useful failure output."""

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return
    print("$ " + " ".join(command), flush=True)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    completed.check_returncode()


def _score_commands(
    *,
    scripts_root: Path,
    eval_path: Path,
    datasets_path: Path,
    error_dir_paths: list[str],
    error_type: str,
    thresholds: tuple[float, ...],
    targets_filename: str,
) -> list[list[str]]:
    return [
        [
            sys.executable,
            (scripts_root / "eval_calc_scores.py").as_posix(),
            f"--error_dir_paths={error_dir_path}",
            f"--eval_path={eval_path.as_posix()}",
            f"--datasets_path={datasets_path.as_posix()}",
            f"--targets_filename={targets_filename}",
            "--visib_gt_min=-1",
            f"--correct_th_{error_type}={threshold}",
        ]
        for error_dir_path in error_dir_paths
        for threshold in thresholds
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    toolkit_root = Path(args.toolkit_root).resolve()
    scripts_root = toolkit_root / "scripts"
    results_path = Path(args.results_path).resolve()
    eval_path = Path(args.eval_path).resolve()
    datasets_path = Path(args.datasets_path).resolve()
    eval_path.mkdir(parents=True, exist_ok=True)
    loaded_alias = os.environ.get("POSETESTBOT_BOP_ADAPTER_LOADED")
    if loaded_alias != args.dataset_alias:
        raise RuntimeError(
            "PoseTestBot custom-dataset adapter was not loaded for this process"
        )
    split_params = dataset_params.get_split_params(
        datasets_path, args.dataset_alias, args.split, None
    )
    if tuple(split_params["im_size"]) != tuple(args.image_size):
        raise RuntimeError("BOP adapter image size does not match the request")

    result_name, _method, dataset, split, split_type, _extension = (
        inout.parse_result_filename(args.result_filename)
    )
    if dataset != args.dataset_alias or split != args.split or split_type is not None:
        raise ValueError("BOP result filename does not match the adapter dataset")
    estimates = inout.load_bop_results(
        results_path / args.result_filename, version="bop19"
    )
    timing_ok, timing_message, timings, timings_available = (
        inout.check_consistent_timings(estimates, "im_id")
    )
    if not timing_ok:
        raise ValueError(timing_message)
    average_time = (
        float(np.mean(list(timings.values())))
        if timings_available and timings
        else -1.0
    )

    started = time.monotonic()
    average_recalls: dict[str, float] = {}
    for error in ERRORS:
        error_type = str(error["type"])
        command = [
            sys.executable,
            (scripts_root / "eval_calc_errors.py").as_posix(),
            "--n_top=-1",
            f"--error_type={error_type}",
            f"--result_filenames={args.result_filename}",
            f"--renderer_type={args.renderer_type}",
            f"--results_path={results_path.as_posix()}",
            f"--eval_path={eval_path.as_posix()}",
            f"--datasets_path={datasets_path.as_posix()}",
            f"--targets_filename={args.targets_filename}",
            "--max_sym_disc_step=0.01",
            "--skip_missing=1",
            f"--num_workers={args.num_workers}",
        ]
        error_dir_paths: list[str] = []
        if error_type == "vsd":
            taus = tuple(float(value) for value in error["taus"])
            command.extend(
                (
                    f"--vsd_deltas={args.dataset_alias}:{args.vsd_delta_mm}",
                    "--vsd_taus=" + ",".join(str(value) for value in taus),
                    "--vsd_normalized_by_diameter=True",
                )
            )
            error_dir_paths = [
                str(
                    Path(result_name)
                    / misc.get_error_signature(
                        "vsd",
                        -1,
                        vsd_delta=args.vsd_delta_mm,
                        vsd_tau=tau,
                    )
                )
                for tau in taus
            ]
        else:
            error_dir_paths = [
                str(Path(result_name) / misc.get_error_signature(error_type, -1))
            ]
        _run(command)

        score_commands = _score_commands(
            scripts_root=scripts_root,
            eval_path=eval_path,
            datasets_path=datasets_path,
            error_dir_paths=error_dir_paths,
            error_type=error_type,
            thresholds=tuple(float(value) for value in error["thresholds"]),
            targets_filename=args.targets_filename,
        )
        print(
            f"Calculating {len(score_commands)} {error_type.upper()} "
            f"score configurations with {args.num_workers} worker(s).",
            flush=True,
        )
        if args.num_workers == 1:
            for score_command in score_commands:
                _run_score(score_command)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                list(executor.map(_run_score, score_commands))

        recalls: list[float] = []
        for error_dir_path in error_dir_paths:
            for threshold in error["thresholds"]:
                score_signature = misc.get_score_signature([threshold], -1)
                score_path = (
                    eval_path / error_dir_path / f"scores_{score_signature}.json"
                )
                scores = inout.load_json(score_path)
                recalls.append(float(scores["recall"]))
        average_recalls[error_type] = float(np.mean(recalls))
        print(
            f"{error_type.upper()} average recall: {average_recalls[error_type]:.6f}",
            flush=True,
        )

    final_scores = {
        "bop19_average_recall_vsd": average_recalls["vsd"],
        "bop19_average_recall_mssd": average_recalls["mssd"],
        "bop19_average_recall_mspd": average_recalls["mspd"],
        "bop19_average_recall": float(
            np.mean(
                (
                    average_recalls["vsd"],
                    average_recalls["mssd"],
                    average_recalls["mspd"],
                )
            )
        ),
        "bop19_average_time_per_image": average_time,
    }
    final_path = eval_path / result_name / "scores_bop19.json"
    inout.save_json(final_path, final_scores)
    print(json.dumps(final_scores, indent=2, sort_keys=True), flush=True)
    print(f"BOP19 evaluation completed in {time.monotonic() - started:.3f}s")
    return final_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkit-root", required=True)
    parser.add_argument("--datasets-path", required=True)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--result-filename", required=True)
    parser.add_argument("--dataset-alias", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, nargs=2, required=True)
    parser.add_argument("--targets-filename", default="test_targets_bop19.json")
    parser.add_argument("--renderer-type", default="vispy")
    parser.add_argument("--vsd-delta-mm", type=float, default=15.0)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.num_workers <= 32:
        raise ValueError("num_workers must be between 1 and 32")
    evaluate(args)


if __name__ == "__main__":
    main()
