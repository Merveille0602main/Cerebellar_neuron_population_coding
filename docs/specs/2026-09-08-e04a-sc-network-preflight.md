# E04a SC Network Preflight

## Scope

Measure whether a vectorized two-layer SC network can support the planned E04a
experiment. This is a temporary throughput and wiring check, not final evidence.

## Design

The preflight uses 16 independent trials. Each has 32 event-driven detailed SC
source cells and two detailed SC output cells. Sources receive one external
excitatory event channel; outputs receive 16 named distributed SC synapses with
four excitatory and twelve inhibitory channels. The source-to-output mapping is
fixed: sources 0-15 drive output 0 and sources 16-31 drive output 1 within each
trial.

The script records detailed source/output events, extracts source spikes into a
16-channel DBNN stimulus plan for every output, and fits a DBNN voltage model
for 10 epochs. Results are stored only below `/tmp/opencode/e04a-preflight`.

## Verification

Check network population/connection counts, finite soma voltages, source event
mapping, split spike coverage, and initialization/training timings. Stop before
the final E04a experiment if the preflight exceeds 30 minutes or reveals a
layout/event-alignment failure.
