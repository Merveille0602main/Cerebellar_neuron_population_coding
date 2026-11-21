# DBNN-GIF Last-Layer Network Example

## Goal

Provide a bounded, executable example that trains a DBNN-GIF approximation of a
small detailed HH target layer, then runs the replacement through `Network.run`.

## Design

- The teacher contains 24 independent HH target cells with two `ExpSyn` input
  channels. A single network run generates all teacher traces.
- `DBNNTrainer` fits the voltage-only `DBNN`; `calibrate_gif()` searches the
  GIF threshold, threshold increment, and threshold decay on holdout spikes.
- The calibrated `DBNNGIF` is switched to recurrent mode and attached with
  `DBNNReduction` to a one-cell replacement network.
- Input data uses interval-start alignment and omits the unused endpoint bin,
  matching the detailed teacher's post-update voltage samples.

## Runtime Budget

The example uses 24 traces of 100 ms at 1 ms resolution, 120 training epochs,
a batch size of 8, and an initialization search limited to two iterations of
population size two. The invocation is externally limited to 10 minutes and
the script rejects an elapsed time beyond nine minutes before reporting output.

## Artifacts

`artifacts/reduction/DBNN_test_network/` contains the generated teacher data,
the calibrated model, and a JSON report. Existing artifacts are replaced only
when the example is explicitly run.
