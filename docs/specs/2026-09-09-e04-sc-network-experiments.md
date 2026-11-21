# E04 SC Network Experiments

## Scope

Implement reproducible E04a, E04b, and E04c SC network experiments. The
implementation must run each case independently, without using validation or
test data to choose a deployment result.

## Cases

- E04a has 32 detailed source cells and two DBNN-GIF output cells with a
  16-channel SC layout.
- E04b has 64 DBNN-GIF source cells and four detailed output cells with a
  16-channel SC layout.
- E04c has 100 DBNN-GIF source cells and eight DBNN-GIF output cells with a
  45-channel SC layout.

Source cells receive fixed external events through their named distributed
synapses. Output cells receive source spike events through their named
distributed synapses. The detailed teacher is always generated before fitting
an asset. Separate source and output assets are trained when a case replaces
that layer.

## Outputs

The runner stores each case below an explicit output root. It writes teacher
and DBNN arrays, trained assets, a metrics report, all timing repetitions, CSV
summary, figures, and a manifest using paths relative to the case directory.
The reported deployment timing excludes training and teacher generation, and
separates network construction, first run, and warm steady-state repetitions.

## Guards

The default profile is a small preflight and never writes to final evidence.
The full profile requires an explicit output root and a time limit. The runner
stops before a case begins when the requested layer is missing its required
asset or when a detailed/deployed layout fingerprint differs.
