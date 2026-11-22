"""Memory estimation and bounded-output execution for long BrainCell runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import resource
import time
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RecordedArrayEstimate:
    category: str
    population: str
    name: str
    sample_shape: tuple[int, ...]
    dtype: str
    bytes_per_step: int
    total_bytes: int
    chunk_bytes: int


@dataclass(frozen=True)
class NetworkOutputEstimate:
    dt_ms: float
    duration_ms: float
    chunk_duration_ms: float
    n_step: int
    n_chunk_step: int
    spike_recording: str
    arrays: tuple[RecordedArrayEstimate, ...]
    total_output_bytes: int
    maximum_chunk_output_bytes: int
    note: str


@dataclass(frozen=True)
class ChunkedRunSummary:
    duration_ms: float
    dt_ms: float
    chunk_duration_ms: float
    n_step: int
    n_chunk: int
    wall_seconds: float
    chunk_wall_seconds: tuple[float, ...]
    output_bytes_processed: int
    persisted_bytes: int
    peak_rss_bytes: int
    spike_counts: Mapping[str, int]
    trace_min: Mapping[str, float]
    trace_max: Mapping[str, float]
    output_directory: str | None


def estimate_network_output_memory(
    network,
    *,
    dt,
    duration,
    chunk_duration,
    spike_recording: str = "population",
) -> NetworkOutputEstimate:
    """Estimate materialized result bytes from initialized sample shapes.

    This is a deterministic lower bound for returned time/trace/spike arrays.
    It excludes cell/runtime state, delivery buffers, compiler caches, JAX
    temporaries, Python objects, filesystem compression, and allocator slack.
    """

    import brainunit as u

    if spike_recording not in ("full", "population", "none"):
        raise ValueError("spike_recording must be 'full', 'population', or 'none'.")
    dt_ms = _time_ms(dt, u)
    duration_ms = _time_ms(duration, u)
    chunk_duration_ms = _time_ms(chunk_duration, u)
    if dt_ms <= 0.0 or duration_ms <= 0.0 or chunk_duration_ms <= 0.0:
        raise ValueError("dt, duration, and chunk_duration must be positive.")
    n_step = len(np.arange(0.0, duration_ms, dt_ms))
    n_chunk_step = len(np.arange(0.0, chunk_duration_ms, dt_ms))
    maximum_chunk_steps = min(n_step, n_chunk_step)

    network.init_state()
    arrays: list[RecordedArrayEstimate] = []
    time_dtype = np.asarray(0.0, dtype=np.float32).dtype
    arrays.append(
        _estimate_array(
            category="time",
            population="",
            name="time_ms",
            sample_shape=(),
            dtype=time_dtype,
            n_step=n_step,
            n_chunk_step=maximum_chunk_steps,
        )
    )
    for population_name, population in network.populations.items():
        for probe_name, sample in population.cell.sample_probes().items():
            sample_array = _magnitude_array(sample)
            arrays.append(
                _estimate_array(
                    category="trace",
                    population=population_name,
                    name=probe_name,
                    sample_shape=sample_array.shape,
                    dtype=sample_array.dtype,
                    n_step=n_step,
                    n_chunk_step=maximum_chunk_steps,
                )
            )
        if spike_recording == "none":
            continue
        spike_array = np.asarray(population.cell.spike.value)
        if spike_recording == "population":
            from braincell.network.delivery import population_spike

            recorded_spike = np.asarray(population_spike(population.cell.spike.value))
        else:
            recorded_spike = spike_array
        arrays.append(
            _estimate_array(
                category="spike",
                population=population_name,
                name="spike",
                sample_shape=recorded_spike.shape,
                dtype=recorded_spike.dtype,
                n_step=n_step,
                n_chunk_step=maximum_chunk_steps,
            )
        )
    return NetworkOutputEstimate(
        dt_ms=dt_ms,
        duration_ms=duration_ms,
        chunk_duration_ms=chunk_duration_ms,
        n_step=n_step,
        n_chunk_step=n_chunk_step,
        spike_recording=spike_recording,
        arrays=tuple(arrays),
        total_output_bytes=sum(item.total_bytes for item in arrays),
        maximum_chunk_output_bytes=sum(item.chunk_bytes for item in arrays),
        note=(
            "Lower bound for materialized outputs only; excludes JAX/compiler/runtime "
            "memory and disk compression. run_chunks preserves one delivery ring state."
        ),
    )


def run_network_in_chunks(
    network,
    *,
    dt,
    duration,
    chunk_duration,
    spike_recording: str = "population",
    output_directory: str | Path | None = None,
    compressed: bool = True,
    event_backend: str = "scatter",
) -> ChunkedRunSummary:
    """Consume a persistent BrainCell chunk iterator without retaining chunks."""

    import brainunit as u

    output_path = None if output_directory is None else Path(output_directory)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=False)

    spike_counts = {name: 0 for name in network.populations}
    trace_min: dict[str, float] = {}
    trace_max: dict[str, float] = {}
    output_bytes = 0
    persisted_bytes = 0
    n_step = 0
    n_chunk = 0
    chunk_wall_seconds: list[float] = []
    started = time.perf_counter()
    chunks = iter(
        network.run_chunks(
            dt=dt,
            duration=duration,
            chunk_duration=chunk_duration,
            event_backend=event_backend,
            spike_recording=spike_recording,
        )
    )
    while True:
        chunk_started = time.perf_counter()
        try:
            result = next(chunks)
        except StopIteration:
            break
        chunk_wall_seconds.append(time.perf_counter() - chunk_started)
        n_chunk += 1
        payload, units = _result_payload(result)
        n_step += int(payload["time_ms"].shape[0])
        output_bytes += sum(value.nbytes for value in payload.values())
        for population_name, spikes in result.spikes.items():
            spike_counts[population_name] += int(np.asarray(spikes).sum())
        for population_name, probes in result.traces.items():
            for probe_name, values in probes.items():
                key = f"{population_name}/{probe_name}"
                array = _magnitude_array(values)
                value_min = float(np.min(array))
                value_max = float(np.max(array))
                trace_min[key] = min(trace_min.get(key, value_min), value_min)
                trace_max[key] = max(trace_max.get(key, value_max), value_max)
        if output_path is not None:
            chunk_path = output_path / f"chunk_{n_chunk:05d}.npz"
            if compressed:
                np.savez_compressed(chunk_path, **payload)
            else:
                np.savez(chunk_path, **payload)
            persisted_bytes += chunk_path.stat().st_size
            units_path = output_path / f"chunk_{n_chunk:05d}_units.json"
            units_path.write_text(json.dumps(units, indent=2, sort_keys=True) + "\n")
            persisted_bytes += units_path.stat().st_size
    wall_seconds = time.perf_counter() - started
    summary = ChunkedRunSummary(
        duration_ms=_time_ms(duration, u),
        dt_ms=_time_ms(dt, u),
        chunk_duration_ms=_time_ms(chunk_duration, u),
        n_step=n_step,
        n_chunk=n_chunk,
        wall_seconds=wall_seconds,
        chunk_wall_seconds=tuple(chunk_wall_seconds),
        output_bytes_processed=output_bytes,
        persisted_bytes=persisted_bytes,
        peak_rss_bytes=_peak_rss_bytes(),
        spike_counts=spike_counts,
        trace_min=trace_min,
        trace_max=trace_max,
        output_directory=None if output_path is None else str(output_path),
    )
    if output_path is not None:
        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")
    return summary


def _estimate_array(
    *,
    category: str,
    population: str,
    name: str,
    sample_shape: tuple[int, ...],
    dtype,
    n_step: int,
    n_chunk_step: int,
) -> RecordedArrayEstimate:
    bytes_per_step = int(np.prod(sample_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if not sample_shape:
        bytes_per_step = np.dtype(dtype).itemsize
    return RecordedArrayEstimate(
        category=category,
        population=population,
        name=name,
        sample_shape=tuple(map(int, sample_shape)),
        dtype=str(np.dtype(dtype)),
        bytes_per_step=bytes_per_step,
        total_bytes=bytes_per_step * n_step,
        chunk_bytes=bytes_per_step * n_chunk_step,
    )


def _result_payload(result) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    import brainunit as u

    payload = {"time_ms": np.asarray(result.time.to_decimal(u.ms))}
    units = {"time_ms": "ms"}
    for population_name, probes in result.traces.items():
        for probe_name, values in probes.items():
            key = f"trace__{population_name}__{probe_name}"
            payload[key] = _magnitude_array(values)
            units[key] = str(getattr(values, "unit", "1"))
    for population_name, values in result.spikes.items():
        key = f"spike__{population_name}"
        payload[key] = np.asarray(values)
        units[key] = "1"
    return payload, units


def _magnitude_array(value) -> np.ndarray:
    return np.asarray(getattr(value, "mantissa", value))


def _time_ms(value, unit_module) -> float:
    return float(np.asarray(value.to_decimal(unit_module.ms), dtype=float).reshape(()))


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB; this project currently targets Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
