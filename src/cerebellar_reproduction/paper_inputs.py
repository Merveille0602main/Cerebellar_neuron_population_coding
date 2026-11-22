"""Convert BSB 3.10.3 spike-generator devices into explicit spike schedules."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PaperMossyInputSchedule:
    """Reproducible mossy schedules generated from one paper simulation config."""

    simulation: str
    duration_ms: float
    device_names: tuple[str, ...]
    source_ids: tuple[int, ...]
    spike_times_ms: Mapping[int, tuple[float, ...]]
    seed: int
    ignored_sd_devices: tuple[str, ...]
    note: str


def build_paper_mossy_input_schedule(
    config_path: str | Path,
    source_mossy_ids: Sequence[int],
    *,
    simulation: str = "stim_on_MFs",
    device_names: Sequence[str] = ("initial_mf", "background"),
    seed: int = 0,
    target_overrides: Mapping[str, Sequence[int]] | None = None,
) -> PaperMossyInputSchedule:
    """Generate explicit schedules using the original BSB 3.10.3 semantics.

    For a noisy device, BSB 3.10.3 calls ``poisson_train`` with mean ISI
    ``interval`` and a window of ``interval * number``.  Its configuration's
    optional ``sd`` field is not consumed.  The original implementation seeds
    each target from system entropy, so an old realization cannot be recovered;
    this converter derives a stable per-device/per-target MT19937 seed instead.
    """

    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    try:
        simulation_config = config["simulations"][simulation]
        devices = simulation_config["devices"]
    except KeyError as exc:
        raise KeyError(
            f"Simulation {simulation!r} or its devices are absent from {config_path}."
        ) from exc

    source_ids = tuple(int(value) for value in source_mossy_ids)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_mossy_ids must be unique.")
    overrides = {
        name: tuple(map(int, values))
        for name, values in (target_overrides or {}).items()
    }
    selected_devices = tuple(str(value) for value in device_names)
    unknown_devices = tuple(name for name in selected_devices if name not in devices)
    if unknown_devices:
        raise KeyError(f"Devices {unknown_devices!r} do not exist in simulation {simulation!r}.")

    combined: dict[int, list[float]] = {source_id: [] for source_id in source_ids}
    ignored_sd: list[str] = []
    for device_index, device_name in enumerate(selected_devices):
        device = devices[device_name]
        if "mossy_fibers" not in device.get("cell_types", ()):  # not our input layer
            continue
        targets = _device_targets(
            device,
            source_ids,
            override=overrides.get(device_name),
        )
        if "spike_times" in device:
            pattern = tuple(float(value) for value in device["spike_times"])
            _validate_generated_pattern(pattern, device_name=device_name)
            for target in targets:
                combined[target].extend(pattern)
            continue

        parameters = device.get("parameters", {})
        start = float(parameters["start"])
        interval = float(parameters["interval"])
        number = int(parameters["number"])
        if interval <= 0.0 or number < 0:
            raise ValueError(
                f"Device {device_name!r} requires interval > 0 and number >= 0."
            )
        if "sd" in parameters:
            ignored_sd.append(device_name)
        noisy = bool(parameters.get("noise", False))
        for target in targets:
            if noisy:
                target_seed = _stable_target_seed(seed, device_index, target)
                rng = np.random.RandomState(target_seed)
                pattern = _bsb3103_poisson_train(
                    interval_ms=interval,
                    duration_ms=interval * number,
                    start_ms=start,
                    rng=rng,
                )
            else:
                pattern = tuple(start + index * interval for index in range(number))
            combined[target].extend(pattern)

    schedules = {
        source_id: tuple(sorted(set(times)))
        for source_id, times in combined.items()
    }
    return PaperMossyInputSchedule(
        simulation=simulation,
        duration_ms=float(simulation_config["duration"]),
        device_names=selected_devices,
        source_ids=source_ids,
        spike_times_ms=schedules,
        seed=int(seed),
        ignored_sd_devices=tuple(ignored_sd),
        note=(
            "BSB 3.10.3 statistical semantics with deterministic replacement seeds. "
            "The publication did not store the system-entropy RNG states, so this is "
            "not the historical random realization. The BSB 3.10.3 generator ignored sd."
        ),
    )


def _device_targets(
    device: Mapping,
    source_ids: Sequence[int],
    *,
    override: Sequence[int] | None,
) -> tuple[int, ...]:
    available = set(source_ids)
    if override is not None:
        requested = tuple(map(int, override))
        unknown = tuple(value for value in requested if value not in available)
        if unknown:
            raise ValueError(f"Input device targets {unknown!r} are not mossy source IDs.")
    elif device.get("targetting") == "by_id":
        requested = tuple(
            value
            for value in map(int, device.get("targets", ()))
            if value in available
        )
    else:
        requested = tuple(source_ids)
    return requested


def _stable_target_seed(base_seed: int, device_index: int, target_id: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), int(device_index), int(target_id)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _bsb3103_poisson_train(
    *,
    interval_ms: float,
    duration_ms: float,
    start_ms: float,
    rng: np.random.RandomState,
) -> tuple[float, ...]:
    """Mirror ``bsb.functions.poisson_train`` from BSB 3.10.3."""

    current = float(start_ms)
    end = float(start_ms + duration_ms)
    values: list[float] = []
    while current <= end:
        current += interval_ms * float(rng.exponential())
        if current > end:
            break
        values.append(current)
    return tuple(values)


def _validate_generated_pattern(pattern: Sequence[float], *, device_name: str) -> None:
    if any(not np.isfinite(value) or value < 0.0 for value in pattern):
        raise ValueError(f"Device {device_name!r} has invalid explicit spike times.")
    if any(later <= earlier for earlier, later in zip(pattern, pattern[1:])):
        raise ValueError(f"Device {device_name!r} spike_times must be strictly increasing.")
