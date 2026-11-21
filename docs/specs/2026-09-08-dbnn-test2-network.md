# DBNN Test2 Network

## Scope

Add `examples/multi_compartment/DBNN_test2_network.ipynb` as an independent
network-level DBNN-GIF experiment. Existing HH and SC notebooks remain
unchanged.

## Design

Each of 1,000 vectorized trials contains four excitatory and three inhibitory
single-compartment HH source cells. Independently seeded two-pulse current
clamps vary first-pulse delay, inter-pulse interval, and amplitude for every
source/trace pair. Their live spike ports connect to one
active, three-branch HH output cell with ten CVs. The output cell has four
distributed excitatory `ExpSyn` inputs with `0.005 uS` weights and three
distributed inhibitory `Exp2Syn` inputs with `0.001 uS` weights. A temporary
three-group teacher probe selected this lowest tested combination because it
produced output spikes without forcing every trace to spike.

The notebook extracts the source spike events from the teacher Network result,
rasterizes them as seven DBNN channels, and trains one DBNN-GIF replacement for
exactly 1,000 epochs with 800/100/100 train/validation/test splits. Training
uses the built-in StepLR with a 200-epoch interval and 0.5 decay, preventing
the learning rate from becoming negligible during the longer run; early stopping
is disabled for the long-run convergence check. It evaluates the
replacement with the same recorded source events, so the detailed teacher and
reduced runtime have equivalent input spike trains.

## Verification

Compile notebook code cells, verify source and output spike coverage before
training, then execute the notebook. Check its embedded output has no errors,
the saved teacher arrays have the expected shapes, and the metrics report is
written below `artifacts/reduction/DBNN_test2_network/`.
