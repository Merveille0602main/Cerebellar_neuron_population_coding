#!/usr/bin/env python3
"""Export reproducible mossy spike schedules from the paper's BSB config."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from cerebellar_reproduction import build_paper_mossy_input_schedule


ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = ROOT.parent / "deschepper-etal-2022-source/configs/balanced.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--spec", type=Path, default=ROOT / "data/export/balanced")
    parser.add_argument("--simulation", default="stim_on_MFs")
    parser.add_argument("--devices", nargs="+", default=["initial_mf", "background"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.spec / "connectivity/mossy_to_glomerulus.npz") as data:
        source_ids = np.unique(data["cell_pairs"][:, 0]).astype(int).tolist()
    schedule = build_paper_mossy_input_schedule(
        args.config,
        source_ids,
        simulation=args.simulation,
        device_names=args.devices,
        seed=args.seed,
    )
    payload = asdict(schedule)
    payload["spike_times_ms"] = {
        str(source_id): list(times)
        for source_id, times in schedule.spike_times_ms.items()
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
