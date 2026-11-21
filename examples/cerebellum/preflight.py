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

"""Run a one-Purkinje-cell BrainCell circuit preflight."""

from __future__ import annotations

import json

import brainunit as u
import numpy as np

from braincell import Network, mech, observe
from braincell.filter import at
from examples.neuron_compare.cell.bc_ma2025.bc_braincell import BC
from examples.neuron_compare.cell.bc_ma2025.parameters import load_bc25_params
from examples.neuron_compare.cell.grc_ma2020.grc_braincell import GrC
from examples.neuron_compare.cell.grc_ma2020.parameters import load_grc20_params
from examples.neuron_compare.cell.io_zh2019.io_braincell import IO
from examples.neuron_compare.cell.io_zh2019.parameters import load_io19_params
from examples.neuron_compare.cell.pc_ma2024.parameters import load_pc24_params
from examples.neuron_compare.cell.pc_ma2024.pc_braincell import PC
from examples.neuron_compare.cell.sc_ma2021.parameters import load_sc21_params
from examples.neuron_compare.cell.sc_ma2021.sc_braincell import SC

# Branch class, conductance (uS), reversal (mV); illustrative, not BSB parameters.
_AFFERENTS = {
    "GrC": ("dendrite", 0.006, 0.0),
    "BC": ("soma", 0.020, -75.0),
    "SC": ("dendrite", 0.025, -75.0),
    "IO": ("dendrite", 0.025, 0.0),
}


def _build():
    net = Network(name="cerebellar_pc_preflight")
    cells = {
        "GrC": GrC(params=load_grc20_params()).build().cell,
        "BC": BC(params=load_bc25_params()).build().cell,
        "SC": SC(params=load_sc21_params()).build().cell,
        "IO": IO(params=load_io19_params()).build().cell,
        "PC": PC(params=load_pc24_params()).build().cell,
    }
    soma = at("soma", 0.5)
    for name, cell in cells.items():
        cell.V_th = 0.0 * u.mV
        cell.place(soma, mech.StateProbe(name="soma_v"))
        if name != "PC":
            # Evoke presynaptic activity for routing validation, not a basal activity protocol.
            amp = {"GrC": 0.20, "BC": 0.05, "SC": 0.05, "IO": 0.05}[name]
            cell.place(soma, mech.CurrentClamp(delay=10 * u.ms, durations=30 * u.ms, amplitudes=amp * u.nA))
        net.add_population(name, cell)

    pc = cells["PC"]
    dendrites = tuple(branch.name for branch in pc.morpho.branches if branch.type == "dendrite")
    if not dendrites:
        raise ValueError("PC preflight requires a dendritic morphology.")
    rows = {}
    for pre, (target_type, weight, reversal) in _AFFERENTS.items():
        branch = "soma" if target_type == "soma" else dendrites[0]
        name = f"{pre}_to_PC"
        connection = net.connect(
            name,
            source=net.populations[pre].event_outputs["spike"],
            target=net.populations["PC"],
            locations=at(branch, 0.5),
            synapse=mech.Synapse("ExpSyn", name=name, tau=2.0 * u.ms, e=reversal * u.mV),
            weight=weight * u.uS,
            delay=0.5 * u.ms,
        )
        placements = connection.synapse
        branch_id = int(placements.branch_id.item())
        resolved_branch = pc.morpho.branches[branch_id]
        x = float(placements.branch_x.item())
        if resolved_branch.name != branch or resolved_branch.type != target_type or x != 0.5:
            raise RuntimeError(f"Unexpected synaptic placement for {name}.")
        pc.record(name, observe.synapse(name=name).state("g"))
        rows[name] = {
            "branch": resolved_branch.name,
            "type": resolved_branch.type,
            "x": x,
            "edges": len(connection),
            "weight_uS": weight,
            "reversal_mV": reversal,
        }
    return net, rows


def _run():
    net, rows = _build()
    duration_ms = 100.0
    result = net.run(dt=0.1 * u.ms, duration=duration_ms * u.ms)
    for name, row in rows.items():
        values = result.samples["PC"][name].values
        conductance = np.asarray(values.to_decimal(u.uS))
        if not np.isfinite(conductance).all() or not np.any(conductance > 0.0):
            raise RuntimeError(f"No valid synaptic conductance response for {name}.")
        row["peak_conductance_uS"] = float(conductance.max())
    populations = {}
    for name in net.populations:
        voltage = np.asarray(result.traces[name]["soma_v"].to_decimal(u.mV))
        if voltage.size != 1000 or not np.isfinite(voltage).all():
            raise RuntimeError(f"Invalid soma voltage trace for {name}.")
        series = result.events[name]["spike"]
        times = np.asarray(series.time.to_decimal(u.ms))
        counts = np.asarray(series.count)
        spikes = int(counts.sum())
        # Each population has one cell. Coincident events contribute zero ISI.
        populations[name] = {
            "spikes": spikes,
            "rate_hz": spikes * 1000.0 / duration_ms,
            "mean_isi_ms": float(np.ptp(times) / (spikes - 1)) if spikes > 1 else None,
            "voltage_range_mV": [float(voltage.min()), float(voltage.max())],
            "peak_1ms_rate_hz": float(np.histogram(times, bins=np.arange(101), weights=counts)[0].max() * 1000),
        }
    return {"duration_ms": duration_ms, "projections": rows, "populations": populations}


if __name__ == "__main__":
    print(json.dumps(_run(), indent=2))
