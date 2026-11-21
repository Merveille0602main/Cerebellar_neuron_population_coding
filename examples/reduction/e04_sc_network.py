# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run the E04a/E04b/E04c layered SC DBNN replacement experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import brainstate
import brainunit as u
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import braincell
from braincell import mech
from braincell.filter import at
from braincell.reduction.dbnn import DBNN, DBNNGIF, DBNNReduction, DBNNTrainer
from braincell.reduction.dbnn.layout import build_channel_layout
from braincell.reduction.dbnn.stimulus import SparseEvents, StimulusPlan
from braincell.reduction.dbnn.train import build_fit_mask, evaluate_predictions
from examples.neuron_compare.cell.sc_ma2021.parameters import load_sc21_params
from examples.neuron_compare.cell.sc_ma2021.sc_braincell import SC
from examples.reduction.sc_to_dbnn import ALL_DENDRITES, ALL_KINDS


DT = 1.0 * u.ms
E04_SEED = 20260909
E_DENDRITES_16 = (8, 12, 14, 21)
I_DENDRITES_16 = (8, 12, 14, 21, 22, 23, 26, 27, 28, 31, 32, 34)


@dataclass(frozen=True)
class Case:
    """Specify one E04 network boundary and SC layout."""

    name: str
    source_cells: int
    output_cells: int
    channels: int
    replace_sources: bool
    replace_outputs: bool


CASES = {
    "E04a": Case("E04a", 32, 2, 16, False, True),
    "E04b": Case("E04b", 64, 4, 16, True, False),
    "E04c": Case("E04c", 100, 8, 45, True, True),
}


@dataclass(frozen=True)
class Profile:
    """Bound the teacher-data and training cost for one run."""

    trials: int
    duration_ms: float
    train_trials: int
    validation_trials: int
    epochs: int
    timing_repeats: int


PROFILES = {
    "preflight": Profile(8, 500.0, 5, 1, 10, 3),
    "full": Profile(24, 1000.0, 16, 4, 500, 10),
}


def parse_args(argv=None) -> argparse.Namespace:
    """Parse one explicit E04 case invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="preflight")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=7200.0)
    parser.add_argument("--stage", choices=("train", "full"), default="full")
    return parser.parse_args(argv)


def channel_specification(channels: int) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Return fixed distributed SC channel semantics for 16 or 45 synapses."""
    if channels == 16:
        dendrites = E_DENDRITES_16 + I_DENDRITES_16
        kinds = ("E",) * 4 + ("I",) * 12
    elif channels == 45:
        dendrites, kinds = ALL_DENDRITES, ALL_KINDS
    else:
        raise ValueError(f"E04 supports only 16 or 45 channels, got {channels}.")
    names = tuple(f"{kind.lower()}_{index:02d}" for index, kind in enumerate(kinds))
    weights = np.asarray([0.0002 if kind == "E" else 0.0005 for kind in kinds], dtype=np.float32)
    return tuple(dendrites), tuple(kinds), names, weights


def build_sc(pop_size, name: str) -> tuple[braincell.Cell, tuple[int, ...]]:
    """Build a detailed SC cell population and its dendrite branch IDs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"from_points\(\) produced .* zero-length segment")
        assembly = SC(params=load_sc21_params(), pop_size=pop_size, name=name).build()
    return assembly.cell, tuple(branch.index for branch in assembly.morpho.branches if branch.type == "dendrite")


def add_channel_synapses(cell, dendrite_ids, dendrites, kinds, names) -> tuple[mech.Synapse, ...]:
    """Place the named distributed synapses represented by one DBNN layout."""
    specs = tuple(
        mech.Synapse(
            "Exp2Syn",
            name=name,
            tau1=0.5 * u.ms,
            tau2=(2.0 if kind == "E" else 5.0) * u.ms,
            e=(0.0 if kind == "E" else -80.0) * u.mV,
        )
        for name, kind in zip(names, kinds)
    )
    for dendrite, spec in zip(dendrites, specs):
        cell.place(at(dendrite_ids[dendrite], 0.5), spec)
    return specs


def build_layout(channels: int):
    """Build the canonical layout shared by source and output SC assets."""
    dendrites, kinds, names, weights = channel_specification(channels)
    cell, branch_ids = build_sc((1,), "e04_layout_template")
    specs = add_channel_synapses(cell, branch_ids, dendrites, kinds, names)
    layout = build_channel_layout(
        cell,
        specs,
        reference_weights_us=tuple(float(value) for value in weights),
        training_ranges=((0.0, 1.0),) * channels,
    )
    return layout, dendrites, kinds, names, weights


def _external_events(size: int, channels: int, steps: int, seed: int) -> tuple[braincell.EventSequence, ...]:
    """Generate deterministic per-channel external event populations."""
    random = brainstate.random.RandomState(seed)
    sources = []
    for channel in range(channels):
        rate_hz = 25.0 + 5.0 * (channel % 4)
        raster = np.asarray(random.uniform(0.0, 1.0, size=(size, steps))) < rate_hz / 1000.0
        cell_ids, bins = np.nonzero(raster)
        sources.append(
            braincell.EventSequence(
                size=size,
                events=braincell.EventTable(source_index=cell_ids, time=(bins + 1) * u.ms),
                name=f"external_{channel:02d}",
            )
        )
    return tuple(sources)


def is_deployment(case: Case, source_model, output_model) -> bool:
    """Validate one asset boundary and identify a deployed rather than teacher run."""
    deployed = source_model is not None or output_model is not None
    if not deployed:
        return False
    if case.replace_sources != (source_model is not None):
        raise ValueError("Source asset presence must match the E04 source replacement boundary.")
    if case.replace_outputs != (output_model is not None):
        raise ValueError("Output asset presence must match the E04 output replacement boundary.")
    return True


def build_network(case: Case, profile: Profile, *, source_model=None, output_model=None):
    """Build a trial-major detailed or reduced E04 network with fixed events."""
    is_deployment(case, source_model, output_model)
    layout, dendrites, kinds, names, weights = build_layout(case.channels)
    source_size = profile.trials * case.source_cells
    output_size = profile.trials * case.output_cells
    network = braincell.Network(f"{case.name.lower()}_network", seed=E04_SEED)
    sources, source_branches = build_sc((source_size,), "sources")
    outputs, output_branches = build_sc((output_size,), "outputs")
    source_specs = add_channel_synapses(sources, source_branches, dendrites, kinds, names)
    output_specs = add_channel_synapses(outputs, output_branches, dendrites, kinds, names)
    if source_model is not None:
        if source_model.layout.fingerprint != layout.fingerprint:
            raise ValueError("Source DBNN asset layout does not match the E04 layout.")
        sources.add_reduction("dbnn", DBNNReduction(source_model))
        sources.use_model("dbnn")
    if output_model is not None:
        if output_model.layout.fingerprint != layout.fingerprint:
            raise ValueError("Output DBNN asset layout does not match the E04 layout.")
        outputs.add_reduction("dbnn", DBNNReduction(output_model))
        outputs.use_model("dbnn")
    for cell, model in ((sources, source_model), (outputs, output_model)):
        if model is None:
            cell.soma.record("voltage", braincell.observe.state("v"))
        else:
            cell.record("voltage", braincell.observe.output("voltage"))
    source_population = network.add_population("sources", sources)
    output_population = network.add_population("outputs", outputs)
    external = _external_events(source_size, case.channels, int(profile.duration_ms), E04_SEED)
    positions = np.arange(source_size, dtype=np.int64)
    for channel, event_source in enumerate(external):
        network.add_population(event_source.name, event_source)
        network.connect(
            f"{event_source.name}_to_sources",
            source=event_source[positions],
            synapse=source_population.synapses[source_specs[channel].name][positions],
            weight=weights[channel] * u.uS,
        )
    source_ids = np.arange(source_size, dtype=np.int64)
    output_ids = np.arange(output_size, dtype=np.int64)
    trial_ids = output_ids // case.output_cells
    output_local = output_ids % case.output_cells
    for channel, spec in enumerate(output_specs):
        # E04c shares the 100 source cells within a trial across output channels.
        local_source = (output_local * case.channels + channel) % case.source_cells
        selected_sources = trial_ids * case.source_cells + local_source
        network.connect(
            f"sources_to_outputs_{channel:02d}",
            source=source_population.event_outputs["spike"][selected_sources],
            synapse=output_population.synapses[spec.name][output_ids],
            weight=weights[channel] * u.uS,
        )
    return network, layout


def _events_to_plan(events, *, population_size: int, channels: int, duration_ms: float, layout, seed_base: int, label: str):
    """Convert named EventSequence records into a DBNN stimulus plan."""
    trace, channel, times = [], [], []
    for channel_id in range(channels):
        record = events[f"external_{channel_id:02d}"]["event"]
        trace.append(np.asarray(record.source_id, dtype=np.int64))
        channel.append(np.full(len(record.time), channel_id, dtype=np.int64))
        times.append(np.asarray(record.time.to_decimal(u.ms), dtype=np.float32))
    return StimulusPlan(
        layout_fingerprint=layout.fingerprint,
        n_traces=population_size,
        events=SparseEvents(
            trace_id=np.concatenate(trace), channel_id=np.concatenate(channel), time_ms=np.concatenate(times)
        ),
        channel_weights=np.ones((population_size, channels), dtype=np.float32),
        duration_ms=duration_ms,
        seeds=tuple(range(seed_base, seed_base + population_size)),
        protocol_labels=(label,) * population_size,
        split="teacher",
    )


def _source_output_plan(events, *, case: Case, profile: Profile, layout):
    """Map source spikes to each output cell's named input channels."""
    record = events["sources"]["spike"]
    source_ids = np.asarray(record.source_id, dtype=np.int64)
    trace, channel, times = [], [], []
    for output in range(profile.trials * case.output_cells):
        trial, local_output = divmod(output, case.output_cells)
        for channel_id in range(case.channels):
            local_source = (local_output * case.channels + channel_id) % case.source_cells
            source = trial * case.source_cells + local_source
            selected = source_ids == source
            trace.append(np.full(np.count_nonzero(selected), output, dtype=np.int64))
            channel.append(np.full(np.count_nonzero(selected), channel_id, dtype=np.int64))
            times.append(np.asarray(record.time.to_decimal(u.ms), dtype=np.float32)[selected])
    return StimulusPlan(
        layout_fingerprint=layout.fingerprint,
        n_traces=profile.trials * case.output_cells,
        events=SparseEvents(
            trace_id=np.concatenate(trace), channel_id=np.concatenate(channel), time_ms=np.concatenate(times)
        ),
        channel_weights=np.ones((profile.trials * case.output_cells, case.channels), dtype=np.float32),
        duration_ms=profile.duration_ms,
        seeds=tuple(range(8000, 8000 + profile.trials * case.output_cells)),
        protocol_labels=(f"{case.name}_output",) * (profile.trials * case.output_cells),
        split="teacher",
    )


def _trial_data(inputs, voltage_mv, population_per_trial: int, profile: Profile):
    """Build disjoint trial-major DBNN train, validation, and test partitions."""
    if inputs.shape[0] != voltage_mv.shape[0] or inputs.shape[-1] != voltage_mv.shape[-1]:
        raise ValueError(
            "DBNN inputs and teacher voltage must have matching trace/time shapes, "
            f"got inputs={inputs.shape} and voltage={voltage_mv.shape}."
        )
    trial_ids = {
        "train": np.arange(profile.train_trials),
        "validation": np.arange(profile.train_trials, profile.train_trials + profile.validation_trials),
        "test": np.arange(profile.train_trials + profile.validation_trials, profile.trials),
    }
    result = {}
    for name, ids in trial_ids.items():
        rows = np.concatenate([np.arange(item * population_per_trial, (item + 1) * population_per_trial) for item in ids])
        targets = voltage_mv[rows]
        spikes = tuple(np.flatnonzero((row[1:] >= 0.0) & (row[:-1] < 0.0)) + 1 for row in targets)
        result[name] = {"inputs": inputs[rows], "targets": targets, "spike_times_ms": spikes, "indices": rows}
    return result


def combine_partitions(source_data: dict, output_data: dict) -> dict:
    """Combine structurally identical source/output partitions for one shared asset."""
    combined = {}
    for partition in ("train", "validation", "test"):
        source = source_data[partition]
        output = output_data[partition]
        if source["inputs"].shape[1:] != output["inputs"].shape[1:]:
            raise ValueError("Shared DBNN asset inputs must have matching channel and time dimensions.")
        combined[partition] = {
            "inputs": np.concatenate((source["inputs"], output["inputs"]), axis=0),
            "targets": np.concatenate((source["targets"], output["targets"]), axis=0),
            "spike_times_ms": source["spike_times_ms"] + output["spike_times_ms"],
        }
    return combined


def rasterize_events_numpy(plan: StimulusPlan, *, dt_ms: float = 1.0) -> np.ndarray:
    """Rasterize an interval-start plan on host memory for bounded E04 batches."""
    intervals = int(round(plan.duration_ms / dt_ms))
    if intervals <= 0 or not np.isclose(intervals * dt_ms, plan.duration_ms):
        raise ValueError("E04 plan duration must be a positive integer multiple of dt_ms.")
    result = np.zeros((plan.n_traces, plan.n_channels, intervals), dtype=np.float32)
    bins = np.floor(np.asarray(plan.events.time_ms) / dt_ms + 1e-9).astype(np.int64)
    selected = (bins >= 0) & (bins < intervals)
    np.add.at(
        result,
        (plan.events.trace_id[selected], plan.events.channel_id[selected], bins[selected]),
        plan.channel_weights[plan.events.trace_id[selected], plan.events.channel_id[selected]],
    )
    return result


def train_asset(layout, data, profile: Profile, *, source_fingerprint: str):
    """Fit one DBNN-GIF asset using validation-only GIF calibration."""
    model = DBNN(layout, mode="f", dt=DT)
    model.source_fingerprint = source_fingerprint
    model.input_alignment = "interval-start"
    params = model.get_params()
    params["bias"] = np.asarray(data["train"]["targets"][:, 0].mean(), dtype=np.float32)
    model.set_params(params)
    trainer = DBNNTrainer(
        model,
        learning_rate=0.01,
        lr_step_size=100,
        lr_gamma=0.5,
        gradient_clip=1.0,
        loss="masked_mse",
        seed=42,
        initialization_options={"maxiter": 2, "popsize": 2, "max_pairs": 120, "samples": 16},
    )
    for partition in data.values():
        partition["mask"] = trainer.build_fit_mask(
            partition["targets"], spike_threshold_mv=-20.0, spike_window_pre_ms=3.0, spike_window_post_ms=10.0
        )
    start = time.perf_counter()
    initialization = trainer.search_initialization(data["train"])
    initialization_seconds = time.perf_counter() - start
    start = time.perf_counter()
    history = trainer.fit(data["train"], validation_data=data["validation"], epochs=profile.epochs, batch_size=8)
    training_seconds = time.perf_counter() - start
    if any(len(row) for row in data["validation"]["spike_times_ms"]):
        gif, gif_report = trainer.calibrate_gif(data["validation"], match_window_ms=10.0)
    else:
        gif = DBNNGIF.from_dbnn(trainer.model)
        gif_report = {"gif": {"status": "not_calibrated_no_validation_spikes"}}
    gif.set_mode("r")
    return gif, {
        "initialization": initialization,
        "initialization_seconds": initialization_seconds,
        "training_seconds": training_seconds,
        "history": history,
        "gif": gif_report,
        "test_metrics": trainer.evaluate(data["test"]),
    }


def _sample_values(result, population: str):
    """Extract one population voltage record into trace-major millivolts."""
    return np.asarray(result.samples[population]["voltage"].values.to_decimal(u.mV), dtype=np.float32).T


def _spike_arrays(result, population: str):
    """Extract sparse population spike records as integer IDs and milliseconds."""
    event = result.events[population]["spike"]
    return np.asarray(event.source_id, dtype=np.int64), np.asarray(event.time.to_decimal(u.ms), dtype=np.float32)


def _spike_rows(ids: np.ndarray, times: np.ndarray, population_size: int) -> tuple[np.ndarray, ...]:
    """Group sparse network spike events into sorted per-cell rows."""
    return tuple(times[ids == cell] for cell in range(population_size))


def _json_value(value):
    """Convert arrays and NumPy scalars into JSON-compatible evidence values."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _run_network(case, profile, *, source_model=None, output_model=None):
    """Build and execute one network, retaining construction and run timings."""
    start = time.perf_counter()
    network, layout = build_network(case, profile, source_model=source_model, output_model=output_model)
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = network.run(dt=DT, duration=profile.duration_ms * u.ms)
    return result, layout, build_seconds, time.perf_counter() - start


def _deployment_metrics(reference_voltage, predicted_voltage, reference_ids, reference_times, predicted_ids, predicted_times, rows):
    """Compute all-sample and masked metrics for one held-out deployed layer."""
    reference_spikes = _spike_rows(reference_ids, reference_times, reference_voltage.shape[0])
    predicted_spikes = _spike_rows(predicted_ids, predicted_times, predicted_voltage.shape[0])
    selected_reference = tuple(reference_spikes[int(row)] for row in rows)
    selected_predicted = tuple(predicted_spikes[int(row)] for row in rows)
    target = reference_voltage[rows]
    prediction = predicted_voltage[rows]
    mask = build_fit_mask(
        target, 1.0, spike_threshold_mv=-20.0, spike_window_pre_ms=3.0, spike_window_post_ms=10.0
    )
    return _json_value({
        "all_sample": evaluate_predictions(
            prediction, target, predicted_spikes=selected_predicted, target_spikes=selected_reference
        ),
        "masked": evaluate_predictions(
            prediction, target, mask=mask, predicted_spikes=selected_predicted, target_spikes=selected_reference
        ),
    })


def _relative_manifest(case_dir: Path, paths: list[Path]) -> dict[str, str]:
    """Hash final artifacts and record only case-relative paths."""
    return {
        str(path.relative_to(case_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists() and path.is_file()
    }


def _write_figures(case_dir: Path, detailed, deployed, timings, case: Case):
    """Write required E04 figures from the held-out network outputs."""
    figures = case_dir / "figures"
    figures.mkdir(exist_ok=True)
    time_axis = np.arange(detailed["output_voltage_mv"].shape[-1])
    figure, axis = plt.subplots(figsize=(9, 3))
    axis.plot(time_axis, detailed["output_voltage_mv"][0], label="detailed")
    axis.plot(time_axis, deployed["output_voltage_mv"][0], label="DBNN")
    axis.set(xlabel="time (ms)", ylabel="voltage (mV)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / f"{case.name}_traces.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 3))
    for values, y, label in ((detailed["output_spike_times_ms"], 0, "detailed"), (deployed["output_spike_times_ms"], 1, "DBNN")):
        axis.vlines(values, y, y + 0.8, label=label)
    axis.set(yticks=(0.4, 1.4), yticklabels=("detailed", "DBNN"), xlabel="time (ms)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / f"{case.name}_raster.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 3))
    labels = ("detailed build", "detailed warm", "DBNN build", "DBNN warm")
    values = (timings["detailed_build"], timings["detailed_first"], timings["dbnn_build"], timings["dbnn_first"])
    axis.bar(labels, values)
    axis.set(ylabel="seconds")
    axis.tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    figure.savefig(figures / f"{case.name}_runtime.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 2.5))
    axis.axis("off")
    source_label = "DBNN-GIF" if case.replace_sources else "detailed SC"
    output_label = "DBNN-GIF" if case.replace_outputs else "detailed SC"
    axis.text(0.18, 0.5, f"source layer\n{source_label}", ha="center", va="center", bbox={"boxstyle": "round", "fc": "#d9ead3"})
    axis.annotate("", xy=(0.72, 0.5), xytext=(0.32, 0.5), arrowprops={"arrowstyle": "->"})
    axis.text(0.83, 0.5, f"output layer\n{output_label}", ha="center", va="center", bbox={"boxstyle": "round", "fc": "#cfe2f3"})
    figure.tight_layout()
    figure.savefig(figures / f"{case.name}_architecture.png", dpi=160)
    plt.close(figure)


def _write_accuracy_figure(case_dir: Path, case: Case, metrics: dict):
    """Plot held-out voltage VE and spike F1 for each deployed layer."""
    rows = [(name, values) for name, values in metrics.items() if values is not None]
    figure, axis = plt.subplots(figsize=(6, 3))
    labels = [name for name, _ in rows]
    ve = [values["masked"]["variance_explained"] for _, values in rows]
    f1 = [np.nan if values["masked"]["f1"] is None else values["masked"]["f1"] for _, values in rows]
    positions = np.arange(len(rows))
    axis.bar(positions - 0.18, ve, width=0.36, label="VE")
    axis.bar(positions + 0.18, f1, width=0.36, label="F1")
    axis.set(xticks=positions, xticklabels=labels, ylabel="score")
    axis.legend()
    figure.tight_layout()
    figure.savefig(case_dir / "figures" / f"{case.name}_accuracy.png", dpi=160)
    plt.close(figure)


def run_case(case_name: str, profile_name: str, output_root: Path, time_limit_seconds: float, *, stage: str = "full"):
    """Generate E04 teacher data, fit required assets, deploy, and export evidence."""
    case = CASES[case_name]
    profile = PROFILES[profile_name]
    if profile.train_trials + profile.validation_trials >= profile.trials:
        raise ValueError("The E04 profile must retain at least one held-out trial.")
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive.")
    if stage not in {"train", "full"}:
        raise ValueError("stage must be 'train' or 'full'.")
    case_dir = output_root / case.name
    for directory in (case_dir / "data", case_dir / "models", case_dir / "figures", case_dir / "tables", case_dir / "manifests"):
        directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    teacher_result, layout, teacher_build, teacher_first = _run_network(case, profile)
    source_voltage = _sample_values(teacher_result, "sources")
    output_voltage = _sample_values(teacher_result, "outputs")
    source_plan = _events_to_plan(
        teacher_result.events, population_size=source_voltage.shape[0], channels=case.channels,
        duration_ms=profile.duration_ms, layout=layout, seed_base=4000, label=f"{case.name}_source"
    )
    output_plan = _source_output_plan(teacher_result.events, case=case, profile=profile, layout=layout)
    source_inputs = rasterize_events_numpy(source_plan)
    output_inputs = rasterize_events_numpy(output_plan)
    source_data = _trial_data(source_inputs, source_voltage, case.source_cells, profile)
    output_data = _trial_data(output_inputs, output_voltage, case.output_cells, profile)
    fingerprint = hashlib.sha256(json.dumps({"case": asdict(case), "layout": layout.fingerprint}, sort_keys=True).encode()).hexdigest()
    source_model = source_report = None
    output_model = output_report = None
    shared_report = None
    source_asset_path = output_asset_path = None
    if case.name == "E04c":
        shared_data = combine_partitions(source_data, output_data)
        shared_model, shared_report = train_asset(
            layout, shared_data, profile, source_fingerprint=f"{fingerprint}:shared"
        )
        source_model = output_model = shared_model
        source_asset_path = output_asset_path = case_dir / "models" / f"{case.name}_shared_dbnn_gif.npz"
        shared_model.save(
            source_asset_path, metadata={"source_fingerprint": shared_model.source_fingerprint, "shared_layers": ["source", "output"]}
        )
    elif case.replace_sources:
        source_model, source_report = train_asset(layout, source_data, profile, source_fingerprint=f"{fingerprint}:source")
        source_asset_path = case_dir / "models" / f"{case.name}_source_dbnn_gif.npz"
        source_model.save(source_asset_path, metadata={"source_fingerprint": source_model.source_fingerprint})
    if case.replace_outputs and case.name != "E04c":
        output_model, output_report = train_asset(layout, output_data, profile, source_fingerprint=f"{fingerprint}:output")
        output_asset_path = case_dir / "models" / f"{case.name}_output_dbnn_gif.npz"
        output_model.save(output_asset_path, metadata={"source_fingerprint": output_model.source_fingerprint})
    if stage == "train":
        if case.name != "E04c" or shared_report is None or source_asset_path is None:
            raise ValueError("The train stage is currently reserved for the E04c shared asset measurement.")
        training_report = {
            "case": case.name,
            "profile": asdict(profile),
            "asset": str(source_asset_path.relative_to(case_dir)),
            "shared_layers": ["source", "output"],
            "initialization_seconds": shared_report["initialization_seconds"],
            "training_seconds": shared_report["training_seconds"],
            "gif_calibration": shared_report["gif"],
            "elapsed_seconds": time.monotonic() - start,
        }
        (case_dir / "data" / f"{case.name}_shared_training.json").write_text(
            json.dumps(_json_value(training_report), indent=2, allow_nan=False), encoding="utf-8"
        )
        return _json_value(training_report)
    if time.monotonic() - start > time_limit_seconds:
        raise TimeoutError("E04 stopped before deployment because training exceeded the configured time limit.")
    load_start = time.perf_counter()
    if source_asset_path is not None:
        source_model = DBNNGIF.load(source_asset_path, mode="r", expected_layout=layout, expected_source_fingerprint=source_model.source_fingerprint)
    if output_asset_path is not None:
        output_model = DBNNGIF.load(output_asset_path, mode="r", expected_layout=layout, expected_source_fingerprint=output_model.source_fingerprint)
    model_load_seconds = time.perf_counter() - load_start
    deployed_result, deployed_layout, dbnn_build, dbnn_first = _run_network(
        case, profile, source_model=source_model, output_model=output_model
    )
    if deployed_layout.fingerprint != layout.fingerprint:
        raise RuntimeError("Detailed and deployed E04 layouts differ.")
    repeats = []
    for _ in range(profile.timing_repeats):
        _, _, _, elapsed = _run_network(case, profile, source_model=source_model, output_model=output_model)
        repeats.append(elapsed)
    detailed = {
        "source_voltage_mv": source_voltage,
        "output_voltage_mv": output_voltage,
        "source_spike_ids": _spike_arrays(teacher_result, "sources")[0],
        "source_spike_times_ms": _spike_arrays(teacher_result, "sources")[1],
        "output_spike_ids": _spike_arrays(teacher_result, "outputs")[0],
        "output_spike_times_ms": _spike_arrays(teacher_result, "outputs")[1],
        "time_ms": np.asarray(teacher_result.time.to_decimal(u.ms), dtype=np.float32),
    }
    deployed = {
        "source_voltage_mv": _sample_values(deployed_result, "sources"),
        "output_voltage_mv": _sample_values(deployed_result, "outputs"),
        "source_spike_ids": _spike_arrays(deployed_result, "sources")[0],
        "source_spike_times_ms": _spike_arrays(deployed_result, "sources")[1],
        "output_spike_ids": _spike_arrays(deployed_result, "outputs")[0],
        "output_spike_times_ms": _spike_arrays(deployed_result, "outputs")[1],
        "time_ms": np.asarray(deployed_result.time.to_decimal(u.ms), dtype=np.float32),
    }
    held_out_source_rows = source_data["test"]["indices"]
    held_out_output_rows = output_data["test"]["indices"]
    deployment_accuracy = {
        "source": _deployment_metrics(
            detailed["source_voltage_mv"], deployed["source_voltage_mv"],
            detailed["source_spike_ids"], detailed["source_spike_times_ms"],
            deployed["source_spike_ids"], deployed["source_spike_times_ms"], held_out_source_rows,
        ) if case.replace_sources else None,
        "output": _deployment_metrics(
            detailed["output_voltage_mv"], deployed["output_voltage_mv"],
            detailed["output_spike_ids"], detailed["output_spike_times_ms"],
            deployed["output_spike_ids"], deployed["output_spike_times_ms"], held_out_output_rows,
        ),
    }
    np.savez_compressed(case_dir / "data" / f"{case.name}_teacher.npz", **detailed)
    np.savez_compressed(case_dir / "data" / f"{case.name}_dbnn.npz", **deployed)
    timings = {
        "detailed_build": teacher_build,
        "detailed_first": teacher_first,
        "dbnn_model_load": model_load_seconds,
        "dbnn_build": dbnn_build,
        "dbnn_first": dbnn_first,
        "dbnn_steady_runs": repeats,
        "dbnn_steady_median": float(np.median(repeats)),
    }
    metrics = {
        "case": asdict(case), "profile": asdict(profile), "layout_fingerprint": layout.fingerprint,
        "source_asset": source_report, "output_asset": output_report, "shared_asset": shared_report,
        "timings_seconds": timings,
        "held_out_deployment_accuracy": deployment_accuracy,
        "elapsed_seconds": time.monotonic() - start,
        "held_out_trials": list(range(profile.train_trials + profile.validation_trials, profile.trials)),
    }
    metrics = _json_value(metrics)
    (case_dir / "data" / f"{case.name}_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (case_dir / "data" / f"{case.name}_timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    _write_figures(case_dir, detailed, deployed, timings, case)
    _write_accuracy_figure(case_dir, case, deployment_accuracy)
    summary = case_dir / "tables" / f"{case.name}_summary.csv"
    summary.write_text(
        "case,source_replaced,output_replaced,channels,detailed_first_seconds,dbnn_first_seconds,dbnn_steady_median_seconds\n"
        f"{case.name},{case.replace_sources},{case.replace_outputs},{case.channels},{teacher_first},{dbnn_first},{timings['dbnn_steady_median']}\n",
        encoding="utf-8",
    )
    files = list((case_dir / "data").glob("*")) + list((case_dir / "models").glob("*")) + list((case_dir / "figures").glob("*")) + [summary]
    manifest = {
        "case": asdict(case), "profile": asdict(profile), "seed": E04_SEED,
        "layout_fingerprint": layout.fingerprint, "files_sha256": _relative_manifest(case_dir, files),
    }
    (case_dir / "manifests" / f"{case.name}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    """Run an explicitly selected E04 case and print its final metrics."""
    args = parse_args()
    print(json.dumps(
        run_case(args.case, args.profile, args.output_root, args.time_limit_seconds, stage=args.stage),
        indent=2,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
