# BrainCell cerebellar preflight

Start with one PC and GrC, BC, SC and IO presynaptic populations. Use the existing ASC-only GrC, BC, SC, IO and PC templates; ASC-only GrC does not model PF propagation. Use native Network and ExpSyn with explicit connection rows, and distinguish soma inhibition from dendritic excitation/inhibition. Check section targets, nonempty connections, finite recorded soma voltages and 100 ms event reports. Use explicit smoke-test weights, not canonical physiological parameters. Full BSB spatial placement, UBC, relays, DCNp/DCNi, 26-projection parity and 5000 ms acceptance are separate milestones. Run channel-level comparisons independently of network calibration; don't change rate equations without a failing regression.

## Run and interpret

From the repository root:

```console
JAX_PLATFORMS=cpu python -m examples.cerebellum.preflight
JAX_PLATFORMS=cpu python -m pytest -q examples/cerebellum/preflight_test.py
```

The integration test uses a 100 ms stimulus-evoked protocol with a 0.1 ms time step. It checks actual connected synapse locations, finite soma voltage and a nonzero conductance response for each input. It emits rate, within-cell ISI and peak one-millisecond rate per population. The one-millisecond peak for a single cell is at least 1000 Hz on any bin with one event. Do not use it as a synchrony acceptance criterion for this smoke case. GrC, SC and IO use a plain PC dendrite as a target proxy; this is not a claim about BSB morphology labels, whole-arbor overlaps or mossy/climbing fibre biophysics.

The next network validation stage should use a curated scaffold edge table (including source/target cell IDs, regions, weights, delays and duplicates) if an exact BSB reconstruction is required. Resolve location labels explicitly and fail on missing or unsupported projections, rather than replacing voxel intersections with random adjacency. A separate small BrainCell-only scaffold may use approximate placement and connections if those changes are reported in the topology manifest. The original request enumerates IDs 1–26, with two routes under ID 23; preserve that distinction when tracking the reference.

| Reference IDs | Native preflight coverage | Remaining validation |
| --- | --- | --- |
| 1–7 | — | Mossy/glomerular relays and granular-layer routing |
| 8–9 | One shared GrC → PC test route | Separate ascending-axon and parallel-fiber outputs and edges |
| 10–11 | — | Parallel-fiber input to BC and SC |
| 12 | BC → PC soma midpoint | Scaffold placement and convergence |
| 13 | SC → PC dendrite midpoint | Scaffold placement, duplicate hook and convergence |
| 14–18 | — | Interneuron recurrence and UBC routes |
| 19–21 | — | PC and mossy inputs to DCNp/DCNi |
| 22 | One IO → PC dendritic edge | Fixed-indegree behavior with multiple eligible IO and PC cells |
| 23–26 | — | IO feedback, interneuron collaterals and DCN routes |


