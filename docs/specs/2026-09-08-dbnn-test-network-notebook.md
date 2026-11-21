# DBNN Test Network Notebook

## Scope

Add an unexecuted notebook version of `examples/multi_compartment/DBNN_test_network.py`.
The source script remains available and unchanged.

## Design

The notebook retains the script's model configuration, deterministic seeds, train/validation/test split, nine-minute budget, and artifact layout. The teacher has ten excitatory `Exp2Syn` inputs (`tau1=0.5 ms`, `tau2=5.0 ms`, and `e=0 mV`), with alternating weights of `0.08` and `0.05 uS` and event rates of `45` and `30 Hz`. It generates 120 traces, split into 80 training, 20 validation, and 20 test examples. It derives the repository root from the notebook's working directory because `__file__` is unavailable in Jupyter.

The workflow is split into setup, network helpers, teacher-data generation, DBNN-GIF training, replacement runtime, and persistence cells. Execution outputs and artifact files are not committed.

## Verification

Validate the new `.ipynb` with `nbformat` and compile each Python code cell.
