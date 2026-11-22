#!/usr/bin/env python3
"""Build the exact published cell-level topology in BrainCell."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import braincell

from cerebellar_reproduction.braincell_adapter import add_edge_sets, add_populations
from cerebellar_reproduction.validation import validate_export


@dataclass
class TopologyOnlyCell:
    """Size-only placeholder; replace with a runnable BrainCell cell model."""

    pop_size: tuple[int, ...]


def build(spec_path: Path) -> braincell.Network:
    manifest = validate_export(spec_path)
    cells = {
        name: TopologyOnlyCell((info["n_cell"],))
        for name, info in manifest["populations"].items()
    }
    network = braincell.Network(name="deschepper-etal-2022-topology")
    add_populations(network, spec_path, cells)
    add_edge_sets(network, spec_path)
    return network


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    network = build(parser.parse_args().spec)
    print(network)
    print(f"Total unique edges: {sum(edges.n_edge for edges in network.edge_sets.values())}")


if __name__ == "__main__":
    main()
