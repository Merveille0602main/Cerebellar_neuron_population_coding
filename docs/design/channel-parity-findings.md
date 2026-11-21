# -*- coding: utf-8 -*-
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

# Cerebellar channel parity findings

Date: 2026-11-21. Scope: the seven cerebellum cell templates under
`examples/neuron_compare/cell`. Question: which ion channel explains the
whole-cell NEURON/BrainCell membrane-potential difference, and which part of
that difference is a mechanism defect rather than numerics.

## Method

Two complements to the existing isolated `channel_no_conc` harness:

1. **Leave-one-out.** Build the debug assembler with one toggle disabled on
   both sides and repeat the notebook current clamp. A large drop in error
   identifies a channel whose presence drives the divergence.
2. **Leak-plus-one.** Enable only `leak` plus one channel (plus `cdp` for
   every calcium-using channel, so both sides carry a calcium pool). This
   removes regenerative amplification and isolates the mechanism itself.

Protocols follow each cell's `run.ipynb`; precision 32 for the default runs
and 64 for the control.

## Channel attribution

PC, 20 ms, 0.5 nA, leak-only floor 0.001264 mV:

| Leak + | mae (mV) | max (mV) |
|---|---:|---:|
| cav32 + cdp | 0.001821 | 0.002684 |
| nav + cdp | 0.002557 | 0.096784 |
| kv1p1 | 0.001044 | 0.001751 |
| kv4p3 | 0.000277 | 0.000689 |
| hcn1 | 0.002320 | 0.003229 |
| kca1p1 + cdp | 0.001636 | 0.002289 |

GoC, 100 ms, 0.2 nA, leak-only floor 0.000464 mV:

| Leak + | mae (mV) | max (mV) |
|---|---:|---:|
| kca1p1 + cdp | 0.002498 | 0.003920 |
| kca2p2 + cdp | 0.001610 | 0.002703 |
| hcn2 | 0.001501 | 0.002039 |
| hcn1 | 0.000432 | 0.000801 |
| cahva + cdp | 0.000625 | 0.012197 |
| kv4p3 | 0.001680 | 0.002173 |

No channel exceeds the leak-only floor by more than a few 1e-3 mV, and the
floor itself is float32 round-off: PC drops from 0.001264 to 0.000622 mV when
only `dt` goes from 0.1 to 0.01 ms.

Cav3.2 initially looked like the single culprit at 0.019 mV. That was a harness
artifact: with `cdp` disabled the BrainCell side had no calcium ion while
NEURON still evaluated its Nernst `carev`. Re-running with the pool enabled
gives 0.0018 mV, in line with the floor.

## Whole-cell decomposition

| Cell | full mae | full max | away from spikes (>2 ms) | spikes |
|---|---:|---:|---:|---:|
| PC (float32) | 0.0129 | 0.468 | 0.0020 / 0.0039 | 2 / 2 |
| PC (float64) | 0.0014 | 0.070 | — | 2 / 2 |
| GoC (float32) | 0.114 | 3.846 | — | 6 / 6 |
| GoC (float64) | 0.0105 | 0.510 | — | 6 / 6 |

GoC's 3.8 mV excursion is the largest single number in the suite and it sits in
the last inter-spike interval, after 85 ms of near-threshold integration. At
float64 the same protocol gives 0.0105 mV mean, 0.510 mV max, and peak voltages
that agree to 3e-4 mV. The difference is accumulated float32 round-off
amplified by a cell hovering near threshold, not a mechanism error. Spike
counts and spike times agree in both precisions to sub-millisecond.

## Defects fixed

- DCN CaHVA/CaLVA evaluated `0/0` at 0 mV; replaced by the analytic
  `exprel` limit, current and slope finite at zero.
- Cav3.1/Cav3.3 forward current was finite at 0 mV but the autodiff pass
  produced NaN through the unselected branch; the denominator is now guarded.
  Forward formula and branch selection unchanged.

Both have regression tests in `braincell/channel/calcium_test.py`.

## Dynamic calcium coupling

Same current clamp, leak plus one calcium mechanism and the `cdp` pool on
both sides, `Ci` recorded at the somatic CV against NEURON `cai`:

| Cell | protocol | cai mae (mM) | cai max (mM) | v mae (mV) |
| --- | --- | ---: | ---: | ---: |
| PC (leak + cdp + Cav3.2) | 0.5 nA, 30 ms, dt 0.1 | 1.3e-7 | 8.2e-6 | 7.2e-7 |
| GoC (leak + cdp + CaHVA) | 0.2 nA, 120 ms, dt 0.01 | 5.3e-7 | 1.9e-5 | 4.3e-7 |

Both pools reproduce the time course of free calcium, not just the resting
value. PC reaches a 1.5x peak above baseline; GoC reaches a 100x peak
(2.1e-5 to 3.2e-3 mM), and the two sides track each other through it.

### Open: GoC pool destabilizes at dt 0.1 under strong, sustained calcium influx

With the same GoC case at the notebook step (dt 0.1 ms) BrainCell returns NaN
while NEURON stays finite. The reproducer is leak + pool + CaHVA only, which
holds the soma near +97 mV for the last 40 ms of a 100 ms trace; the standard
`goc_ma2020/run.ipynb` protocol never reaches this regime because the
potassium channels repolarize the cell.

The failure chain is now pinned down:

1. At t = 53.7 ms exactly one compartment's free calcium leaves the physical
   domain: `Ci` = **-5.31e-6 mM** (baseline 4.5e-5 mM), in an axon-AIS shell.
   The pool integrates `Ci` with backward Euler on a 23-species reaction
   system; backward Euler is L-stable but not positivity preserving, and when
   the calcium current reverses to a large outward efflux during the plateau,
   one step's efflux exceeds the standing `Ci`. Smaller dt shrinks the
   excursion, so dt 0.05 still fails (-7.5e-5 mM, t = 60.4 ms) and dt 0.01
   runs clean.
2. That same step's membrane voltage is already NaN, because
   :func:`braincell.ion._base._nernst` evaluates
   :math:`E = RT/(zF)\,\ln(C_o/C_i)` with no guard. `ln` of a negative
   concentration is NaN, `eca` becomes NaN, and `CaHVA` is an *ohmic* calcium
   current that reads `eca`, so the NaN enters the membrane equation in the
   same step. Only then does `Ci` itself go NaN, spreading from 1 to 90
   compartments within two steps.
3. Isolation matrix at dt 0.1: leak + CaHVA (no pool) is finite, because `Ci`
   is never updated; leak + pool (no CaHVA) is finite, `Ci` >= 3.85e-6 mM;
   the two together produce the negative `Ci` and the NaN. The NaN therefore
   needs the pool/channel coupling, not either half.

Raising `default_substeps` to 10 changes nothing: the failing sample is
bit-identical (-5.31227433e-06 mM at step 537), which means the substep
schedule is not reaching this update path -- worth a separate look, since
`KineticIon.make_integration` is supposed to divide `dt` by `substeps`.

A positivity clamp on `Ci` (or a NaN-safe Nernst) would stop the NaN, but it
would only hide the under-resolved efflux. The real fix is either a
positivity-preserving update for the pool's source term or a documented
maximum step for this pool at high calcium drive.

## Change applied

The two remaining float32 comparison notebooks, `pc_ma2024/run.ipynb` and
`goc_ma2020/run.ipynb`, were switched to `precision=64` so every template in
the suite compares at the same precision. Cell models, mechanisms, protocols
and morphology are untouched. Expected effect from the controls above: PC
0.0129 -> 0.0014 mV mean and 0.468 -> 0.070 mV max; GoC 0.114 -> 0.0105 mV mean
and 3.846 -> 0.510 mV max.

## Voltage-clamp I-V parity of the GHK channels

A clean test of the "permeability instead of conductance" mechanisms: hold the
section with a VClamp, let the gates reach steady state, and compare current
density at 29 voltages from -100 to +40 mV, `Ci`=1e-4 mM, `Co`=2 mM, 36 C.

Read at `t=0` after `finitialize(V)`; a settled VClamp sits about
2e-4 mV off target, which is enough to contaminate the comparison at the 1e-8
level and is the reason an earlier version of this table read 2.5e-6 for the
frozen row.

| Channel | permeability | max abs current error | peak current | cause |
| --- | ---: | ---: | ---: | --- |
| Cav2.1 (PC, unfrozen) | 2.2e-4 cm/s | 5.76e-6 mA/cm2 | 0.0908 | constant set |
| Cav2.1 (PC, frozen) | 2.2e-4 cm/s | 1.4e-17 | 0.0908 | exact |
| Cav3.1 (PC) | 2.5e-4 cm/s | 7.8e-10 | 0.0055 | exact |
| Cav3.1 (GoC) | 1.0e-4 cm/s | 3.4e-10 | 0.0022 | exact |
| Cav2.1 (SC) | 2.2e-4 cm/s | 5.76e-6 | 0.0908 | constant set |

The 5.76e-6 on the unfrozen classes is 6.3e-5 relative and comes entirely from
the constant set: `ghk_flux` uses CODATA F/R with 273.15 K, the NEURON build
uses 9.6485e4, 8.3145 and `273.19 + celsius`. Re-evaluating either class with
the other's constants collapses the difference to 1e-17, so no formula, unit or
sign error is involved. The classes whose mod hardcodes its own constants
(Cav3.1, and the frozen Cav2.1 path) already use a matching helper and agree
exactly.

No GHK channel exceeds 1e-5 mA/cm2 in a voltage clamp, so neither the
constant-field formula nor the mS-vs-cm/s parameter convention is a source of
systematic bias. At the 1e-5 level the only residual is the NEURON-vs-CODATA
constant set on the classes that share `ghk_flux`. The isolated suite's larger numbers
(Cav2.1 0.601 mV, Cav3.1 0.166 mV) come from its current-clamp protocol with
leak disabled, where a sub-percent transient current difference integrates into
millivolts; they are not reproducible in a voltage clamp at the same
permeability.

### What `max abs error` measures, and how it maps to millivolts

Every I-V table in this document reports `max |I_bc(V) - I_nrn(V)|` over the
29 clamp voltages, in mA/cm2: a current-density residual of one mechanism at a
fixed membrane potential. Under a voltage clamp the membrane potential is the
*input*, so a voltage residual is identically zero by construction and cannot
be a metric; the only observable is the current the channel needs to hold that
voltage.

The millivolt-level acceptance test is a different experiment -- a
current-clamp whole-cell run, where whatever residual a channel leaves is
integrated by the membrane, `dV/dt = dI / Cm`. With `Cm` between 1 and
2.5 uF/cm2 (the values these cells use) one mA/cm2 charges at 400-1000 mV/ms:

| residual (mA/cm2) | mV per ms | mV after 100 ms |
| ---: | ---: | ---: |
| 1e-8 | 4e-6 .. 1e-5 | 4e-4 .. 1e-3 |
| 1e-11 | 4e-9 .. 1e-8 | 4e-7 .. 1e-6 |
| 1e-16 | 4e-14 .. 1e-13 | 4e-12 .. 1e-11 |

That is where the target comes from: 1e-3 mV over a 100 ms trace is a
*sustained* residual of about 1e-8 mA/cm2. It also explains the ranking -- the
Kca2.2 rows (1.1e-8) sit exactly at the whole-cell floor, and everything else
has four to eight orders of margin. The conversion is an upper bound, not a
prediction: a residual that changes sign with voltage, or exists only where the
channel carries little current, integrates to far less. The Cav2.1 row is the
example -- 6.5e-6 would be 0.7 mV over 100 ms if it were DC, while the
whole-cell PC trace moves only 6.9e-4 mV, because the difference is relative
and vanishes where the calcium current is small.

## Voltage-clamp I-V parity of the calcium mechanisms

Same harness as below (one mechanism per 10 um x 31.83 um section, `VClamp`,
`t=0` after `finitialize(V)`, 29 voltages -100..+40 mV, `cai` = 1e-4 mM,
`cao` = 2 mM, 36 C), run for all 21 calcium mechanisms of the seven cells.
Both the GHK/permeability classes (`pcabar`/`perm` in cm/s) and the ohmic ones
(`gcabar`/`gbar` in S/cm2) are included; each BrainCell channel was given the
same physical density as the mod, 2.5e-4 cm/s or 10 mS/cm2.

| channel | cells | form | max abs error (mA/cm2) |
| --- | --- | --- | ---: |
| Cav2.1 | PC, SC, BC | GHK | 6.5e-6 |
| Cav3.1 | PC, GoC | GHK | 3.5e-18 |
| Cav3.2 | PC, SC, BC | ohmic | 1.7e-18 |
| Cav3.3 | PC, SC | GHK | 7.4e-8 |
| Cav1.2 | GoC, BC | ohmic | 2.2e-16 |
| Cav1.3 | GoC, BC | ohmic | 5.6e-17 |
| Cav2.3 | GoC | ohmic | 1.7e-18 |
| CaHVA | GoC, GrC | ohmic | 2.1e-17 |
| CaHVA | DCN | GHK | 4.2e-17 |
| CaLVA | DCN | GHK | 1.6e-19 |
| CaL | DCN | ohmic | 4.3e-19 |
| Ca | IO | ohmic | 8.7e-19 |

The Cav2.1 rows are the same 6.3e-5 relative constant-set difference documented
above; every other GHK class is exact. **Cav3.3 is the one open item**: its mod
writes `ica = gCav3_3bar * pcabar * n * n * l * ghk` with a private
`ghk = -0.001*z*F*(cao - cai*exp(w))*w/(exp(w)-1)`, and no single permeability
reproduces its I-V shape (the residual is 99 per cent relative, though only
7.4e-8 mA/cm2 absolute, because the mod's own current at 2.5e-4 cm/s is tiny).
The BrainCell class ports `pcabar` alone and folds the `gCav3_3bar` factor
differently, so this needs a dedicated comparison rather than a scale factor.

Three mods -- `Cav3p3_MA24_PC`, `CaHVA_SU15_DCN` and `CaLVA_SU15_DCN` --
evaluate an unguarded `w/(exp(w)-1)` at exactly 0 mV and return NaN on the
NEURON side; that single point was dropped from the comparison. BrainCell's
`exprel`/`where` guards avoid the singularity.

## Voltage-clamp I-V parity of the Na / K / HCN mechanisms

Same protocol as the GHK table above: one mechanism per 10 um x 31.83 um
section, fixed with `VClamp`, `g_max` set to the *same physical* density on
both sides (10 mS/cm2), 29 voltages from -100 to +40 mV in 5 mV steps, and the
current read at `t=0` after `finitialize(V)`, so every gate starts from its
steady state. `ena` = 60 mV, `ek` = -88 mV, `eh` = -34.4 mV, `cai` =
5e-3 mM (so Kca1.1/Kca2.2 are measurably open), 22 C. All 61 Na/K/HCN
mechanisms reachable from the seven cerebellar cell models were run.

| family | mechanisms | max abs error (mA/cm2) | max relative | largest peak |
| --- | ---: | ---: | ---: | ---: |
| NaV | 8 | 2.1e-11 | 9.5e-9 | 4.4e-3 |
| Na fast (GrC) | 1 | 5.3e-16 | 4.0e-13 | 1.3e-3 |
| Na fast (DCN) | 1 | 2.8e-17 | 2.3e-16 | 0.122 |
| Na persistent (DCN) | 1 | 1.7e-18 | 1.3e-16 | 0.013 |
| Kv | 19 | 2.2e-16 | 5.4e-16 | 1.280 |
| Kdr (DCN, IO) | 3 | 6.7e-16 | 5.2e-16 | 1.280 |
| M-type K | 3 | 2.2e-16 | 1.7e-16 | 1.280 |
| Kir | 4 | 2.1e-17 | 2.5e-16 | 0.084 |
| KCa (BK, SK) | 13 | 1.1e-8 | 1.1e-8 | 1.024 |
| HCN | 7 | 1.1e-16 | 2.1e-16 | 0.735 |

Per cell, the worst mechanism is 1.9e-15 mA/cm2 (PC), 1.1e-8 (GoC, SC, BC,
GrC -- all at Kca2p2), 6.7e-16 (DCN) and 2.1e-11 (IO). Nothing reaches
1e-7 mA/cm2, so the ohmic Na/K/HCN mechanisms carry no systematic bias; the
channels that reach 1.1e-8 (Kca2.2, four cell variants) are Markov schemes whose NEURON side is solved
with `INITIAL ... STEADYSTATE`/`seqinitial` and whose BrainCell side uses the
generator solve, so the residual is of the same order as the other Markov
mechanisms (Nav1p1 1.5e-11, Na_ZH2019 2.1e-11) and far below anything that
matters at the millivolt level.

Three harness traps cost real time here and will bite anyone repeating this:

- **The prebuilt libraries are from NEURON 7.7.** The
  `x86_64/libnrnmech.so` checked into `Cerebellum_mod/<CELL>/channel/` still
  carries the generated-code banner `Created by Language version: 7.7.0`,
  while this environment runs 8.2.6. Loaded by 8.2.6 it silently drops
  `INITIAL ... STEADYSTATE sparse`: a KINETIC sodium mechanism then starts
  with all states at zero and takes ~10 ms to creep to steady state (`ina` at
  `t=0` reads 0 instead of -1.59e-3 mA/cm2). Rebuild with `nrnivmodl` before
  comparing anything at `t=0`; long current-clamp traces are only mildly
  affected, because the error decays away in the first milliseconds. Note
  that `_nrnmech.nrnmech_path` prefers `.libs/libnrnmech.so`, which is the
  same 7.7 binary, so the notebooks load it too.
- **A mod that reads `ki,ko` re-derives `ek`.** `Kv1p5_MA20_GrC.mod`
  declares `USEION k READ ek,ki,ko`, so NEURON recomputes `ek` from the
  concentrations and an assignment to `sec.ek` does not stick (it reads back
  -78.34 mV instead of -88 mV). Read the effective `ek` back after
  `finitialize` and give BrainCell that value, otherwise the channel looks
  12 per cent off over the whole voltage range.
- **A mod with `RANGE ... eh` and no default leaves `eh` at 0 mV.**
  `HCN_SU15_DCN.mod` is such a mod; the cell template sets
  `sec.eh_HCN_SU15_DCN = -45 mV` explicitly, so the parity harness must too,
  otherwise the h-current is compared against E = 0 mV.

## Whole-cell convergence at float64

| dt (ms) | PC max abs (mV) | GoC max abs (mV) | peak agreement |
| ---: | ---: | ---: | --- |
| 0.1 | 0.070 | 0.510 | 1e-4 |
| 0.01 | 1.29e-3 | 8.45e-3 | 1e-6 |
| 0.0025 | 6.90e-4 | 2.70e-3 | 1e-6 |

The remaining error is the spike upstroke sampling: with dV/dt of order
100 mV/ms, a 10 us timing difference already produces 1e-3 mV. Peak voltages
agree to 1e-6, and both spike counts match. A 1e-3 mV max-difference target is
therefore a statement about the time step and the integration scheme, not about
the channels.

## What remains

- Dynamic `cai` and calcium-dependent currents were exercised only through
  voltage; a paired `cai` trace comparison is still open.
- The comparison harnesses run at float32. Any body aiming at 1e-5 mV should
  use float64, otherwise the reported floor is round-off.
- Whole-cell divergence after long near-threshold integration is chaotic; treat
  max-abs over a 100 ms trace as an amplification metric, not a fidelity metric.
## The channel_no_conc harness floor: float32 round-off, not channel kinetics

The `examples/neuron_compare/channel_no_conc` sweep is a free-run comparison:
both sides start from the same `v_init`, receive the same current, and the
recorded `v` / `ix` / gates are differenced. Its error budget is dominated by
two things that are not channel properties, and both have to be removed before
the harness can be used as a 1e-5 acceptance gate.

### 1. The checked-in mechanism libraries were stale

`examples/neuron_compare/Cerebellum_mod/<CELL>/channel/x86_64/libnrnmech.so`
is gitignored, so every checkout carries whatever was built last. The copies
on disk were built in May 2026 and, when loaded by NEURON 8.2.6, silently skip
`INITIAL { SOLVE ... STEADYSTATE sparse }`: every gating variable starts at 0
and relaxes over the first ~10 ms. Measured on `Nav1p6_MA20_GoC` at -65 mV:

| library | `t = 0` `C1` | `t = 0` `I5` | `t = 0` `ina` (mA/cm^2) |
| --- | ---: | ---: | ---: |
| as checked in (May build) | 0.000000 | 0.000000 | 0.0 |
| rebuilt with the active NEURON | 0.681410 | 0.003264 | -1.50e-4 |

Rebuilding all seven libs in place with the active `nrnivmodl` drops the
`nav1p6_ma20_goc` `vinit_celsius` error from 27.1 mV to 0.68 mV. Rebuilding is
therefore mandatory, but it is not sufficient.

### 2. The default run is float32, and this harness amplifies its round-off

`brainstate` sets `jax_enable_x64 = False` on import unless a `precision=64`
context is active, so `braincell` computes in float32. The spacing of float32
at 65 mV is exactly 7.6293945e-06 mV, and that is precisely the voltage
residual of `cav2p3_ma20_goc`, a config whose `g_max_S_cm2` is 0.0, so its
channel current is identically zero and no channel can contribute:

| config / template | max abs dV, float32 | max abs dV, float64 |
| --- | ---: | ---: |
| `cav2p3_ma20_goc` / `vinit_celsius` | 7.629e-06 | 3.089e-10 |
| `cav2p3_ma20_goc` / `dc` | 3.136e-03 | 3.183e-10 |

Free-running dV/dt = dI/Cm closes a positive feedback loop, dI = g dV + dI0,
with loop rate g/Cm of order 1e3 to 1e6 per second. A float32 round-off level
current mismatch of 1e-11 mA/cm^2 therefore grows by 3 to 4 orders of magnitude
inside a 10 to 20 ms trace. That, not kinetics, is what produced the
0.1 to 60 mV errors in the stored May/June batches. Re-running the same configs
under `brainstate.environ.context(precision=64)`:

| config | template | max abs dV, float32 | max abs dV, float64 | max abs dI, float64 | max gate error, float64 |
| --- | --- | ---: | ---: | ---: | ---: |
| `nav1p6_ma20_goc` | vinit | 6.805e-01 | 3.843e-11 | 4.79e-14 | 5.82e-13 |
| `nav1p6_ma20_goc` | dc | 4.376e+00 | 6.844e-10 | 7.94e-12 | 7.85e-12 |
| `cahva_ma20_goc` | dc | 1.784e-01 | 1.332e-10 | 8.00e-14 | 1.77e-12 |
| `kca3p1_ma20_goc` | dc | 2.641e-01 | 4.914e-11 | 9.04e-15 | 4.21e-16 |
| `cav3p2_ri21_sc` | dc | 9.740e-01 | 7.223e-11 | 3.69e-15 | 9.91e-13 |
| `kca2p2_ma20_goc` | dc | 2.756e-01 | 6.411e-06 | 7.42e-10 | 3.08e-09 |
| `kir2p3_ma24_pc` | dc | 3.697e-02 | 5.676e-11 | 2.18e-14 | 6.50e-13 |
| `kv4p3_ma24_pc` | dc | 2.895e-02 | 1.036e-10 | 7.51e-15 | 6.09e-13 |
| `km_ri21_sc` | dc | 1.413e-03 | 9.166e-11 | 2.47e-15 | 9.05e-14 |
| `hcn2_ma20_goc` | dc | 1.405e-03 | 8.530e-11 | 6.54e-16 | 1.73e-14 |

The channels previously filed as "time-constant mismatches" -- `kir2p3`,
`kv4p3`, `KM`, `HCN2`, `CaHVA` -- are all at 1e-10 to 1e-13 once the run is
float64. There was no time-constant defect; the growth-with-time shape of the
error was the feedback loop integrating round-off.

Two genuine residual defects remain in this harness:

- `ac` template, ~1.0e-02 mV on every config. With `g_max = 0` and `dI == 0`
  identically, `dV` still tracks `dV/dt * dt/2` with correlation 0.999992:
  NEURON applies the played sine at `t + dt` while `SineClamp` applies it
  half a step earlier. Fix the stimulus sampling if `ac` is to be used for
  anything.
- `cav2p1_ri21_sc` / `dc`: 0.63 mV and 1.2e-03 mA/cm^2. This one is real; the
  0.05 nA step drives the compartment to +54 mV, where the RI2021 Cav2.1
  activation differs from the mod file.

Acceptance criteria implied by the above:

- `current < 1e-5 mA/cm^2` and `gates < 1e-5` are reachable in this harness and
  are the right per-channel gates.
- `voltage < 1e-5 mV` is a different kind of statement. It is below one float32
  ULP at rest, and in a free run it needs the sustained current residual to be
  below about 1e-9 mA/cm^2 to survive 10 ms of integration. Use it only with
  float64 and only as a slow-drift indicator.

## Aligning the harness with NEURON's arithmetic and sampling

### Precision

NEURON keeps every mechanism state and parameter in a C `double`: the
generated code declares `double* _p`, and a range variable round-trips
exactly (setting `gbar = 0.016000000123456789` reads back bit-identical,
while that value is not representable in float32). BrainCell instead runs
float32 unless told otherwise, because `brainstate` disables
`jax_enable_x64` on import. The harness therefore had a systematic
mismatch of six decimal digits before any channel code ran.

`engine/run.py` now takes `--precision {32,64}` with a default of 64, and
`braincell_runner.run_case` wraps the run in
`brainstate.environ.context(precision=...)`. The behaviour is unchanged for
existing callers because the default lives in `DEFAULT_PRECISION`.

### Sine stimulus sampling

NEURON applies a vector-played `IClamp` amplitude at the *start* of the
step; BrainCell evaluates `SineClamp` at the *midpoint*. Measured on a
channel-free single compartment by differencing the recorded voltage against
`I = C dV/dt`:

| side | current applied for the step ending at `t` | fit residual |
| --- | --- | ---: |
| NEURON | `sin(w (t - dt))` | 8.8e-15 |
| BrainCell | `sin(w (t - dt/2))` | 1.1e-05 |

`stimulus.build_braincell_stimulus` now accepts `dt_ms` and advances the
phase by `-w dt / 2` (`sine_step_phase_shift`), which reproduces NEURON's
sequence to the measurement floor: the fitted delay moves from
`0.4993 dt` to `0.9988 dt` against NEURON's `1.0000 dt`. The shift is
0.0045 rad at 250 Hz and dt = 0.01 ms, but it removes a `1e-02 mV` artifact
that every config was paying on the `ac` template.

Result on the `ac` template, precision 64, all ten configs that previously
sat at ~`1.0e-02 mV`:

| config | max abs dV | max abs dI | max gate error |
| --- | ---: | ---: | ---: |
| `cahva_ma20_goc` | 9.09e-11 | 6.54e-16 | 1.98e-13 |
| `cav3p2_ri21_sc` | 6.22e-11 | 7.30e-15 | 1.79e-12 |
| `cav3p3_ri21_sc` | 7.22e-09 | 2.76e-18 | 2.97e-11 |
| `hcn2_ma20_goc` | 5.57e-11 | 4.25e-16 | 2.01e-16 |
| `kca2p2_ma20_goc` | 3.47e-06 | 5.51e-10 | 3.08e-09 |
| `kca3p1_ma20_goc` | 7.42e-11 | 1.31e-14 | 4.97e-16 |
| `kir2p3_ma24_pc` | 3.76e-11 | 1.55e-14 | 4.35e-13 |
| `km_ri21_sc` | 6.73e-11 | 7.80e-16 | 9.73e-14 |
| `kv4p3_ma24_pc` | 6.58e-11 | 9.48e-16 | 5.26e-13 |
| `cav2p1_ri21_sc` | 7.99e-05 | 1.14e-08 | 2.80e-07 |

### The two remaining residuals both belong to the protocol, not to a channel

`cav2p1_ri21_sc` is the only config left above `1e-5`, and its two
templates fail for two different reasons.

- `dc`: the 0.05 nA step drives the compartment to +54 mV at ~120 mV/ms.
  The error halves exactly when `dt` halves, so it is a first-order
  integrator difference, not a formula difference:

  | dt (ms) | max abs dV | max abs dI | max gate error |
  | ---: | ---: | ---: | ---: |
  | 0.0100 | 6.27e-01 | 1.21e-03 | 4.96e-03 |
  | 0.0050 | 3.10e-01 | 5.96e-04 | 2.42e-03 |
  | 0.0025 | 1.52e-01 | 2.90e-04 | 1.15e-03 |

  A free-run voltage comparison is only first-order accurate in `dt`, and
  its constant is proportional to |dV/dt|. This is why the quiet
  `vinit_celsius` template reaches `1e-10` while a driven step is stuck
  at `1e-1`: the two integrators are different, and no amount of
  precision or alignment changes that.
- `ac`: `7.99e-05 mV`, independent of `dt` (7.993e-05, 7.922e-05,
  7.887e-05 at dt = 0.01, 0.005, 0.0025) and exactly proportional to the
  conductance (3.805e-05 at half, 1.858e-05 at a quarter). A residual that
  scales with the current scale and not with `dt` is a difference in the
  closed-form current expression. It belongs to the same family as the
  Cav2.1 voltage-clamp residual already recorded (6.5e-06 mA/cm^2, CODATA
  versus NEURON constants), not to gating or timing.

### Config fix

`cav2p3_ma20_goc` declared `g_max_S_cm2 = 0.0`, so its channel current
was identically zero and the config compared nothing. The MA2020 GoC soma
carries no Cav2.3 (`parameters.py` leaves the field at 0.0), but the fitted
apical dendrite does (`Optimization_result.txt` `values[3] =
1.28859564935e-03`). That value is now used as the comparison scale and the
deviation is recorded in the config's `meta.note`.

## The full float64 survey: which channels are still not aligned

All 89 configs of `configs/` were re-run with the aligned harness
(`--precision 64`, sine step shift, rebuilt mechanism libraries) and each
config was run through all three templates: 265 config/template rows.
This is the acceptance record for the `1e-5` gate.

### Headline

- 238 of 265 rows have `max abs dI <= 1e-9 mA/cm^2` **and**
  `max gate error <= 1e-7`.
- 22 rows exceed `1e-5` on at least one observable, and they belong to
  **four mechanism families only**: `Cav2.1`, `Cav3.1`, `CaLVA_SU2015_DCN`
  and `Na_ZH2019_IO`.
- Nothing in the `K+` or `HCN` catalogue appears above `1e-9` on current
  or gating. `kca2p2` is the only K channel whose *voltage* trace drifts,
  at `6.4e-06 mV`, while its current and gates are at `7.4e-10` and
  `3.1e-09`.

| config | template | max abs dV (mV) | max abs dI (mA/cm2) | max gate err | fails |
| --- | --- | ---: | ---: | ---: | --- |
| `cav2p1_ma24_pc` | ac | 7.993e-05 | 1.144e-08 | 2.804e-07 | dV |
| `cav2p1_ma24_pc` | dc | 6.270e-01 | 1.213e-03 | 4.956e-03 | dV/dI/gate |
| `cav2p1_ma24_pc` | vinit_celsius | 2.839e-04 | 7.144e-08 | 3.030e-06 | dV |
| `cav2p1_ma25_bc`, `cav2p1_ri21_sc` | identical to `cav2p1_ma24_pc` | | | | |
| `cav3p1_ma20_goc` | ac | 1.804e-01 | 9.853e-05 | 7.940e-04 | dV/dI/gate |
| `cav3p1_ma20_goc` | dc | 1.658e-01 | 8.089e-05 | 3.855e-04 | dV/dI/gate |
| `cav3p1_ma20_goc` | vinit_celsius | 3.894e-02 | 5.758e-06 | 1.696e-04 | dV/gate |
| `cav3p1_ma24_pc`, `cav3p1_test2_pc24` | identical to `cav3p1_ma20_goc` | | | | |
| `calva_su15_dcn` | ac | 9.339e-05 | 1.648e-06 | 1.644e-07 | dV |
| `calva_su15_dcn` | dc | 9.334e-05 | 1.648e-06 | 1.644e-07 | dV |
| `calva_su15_dcn` | vinit_celsius | 3.207e-04 | 2.127e-05 | 2.629e-07 | dV/dI |
| `na_zh19_io` | vinit_celsius | 2.579e-05 | 3.435e-07 | 2.710e-07 | dV |

As a reference for what "aligned" looks like here, the best families sit
at `dV ~ 1e-10`, `dI ~ 1e-15` and gates `~1e-13` (`cahva`, `cav1p2`,
`cav3p2`, `cav2p3`, `ca_zh19_io`, `kir2p3`, `hcn1`, `hcn2`, `km`,
`kv4p3`).

### `Cav2.1`: the config compares the wrong BrainCell class

The three `cav2p1` configs map NEURON `Cav2p1_*` to the **unfrozen**
BrainCell class, e.g. `Cav2p1_MA2024_PC`. That class evaluates the GHK
drive with the generic `ghk_flux` helper (CODATA `F`, `R`), while its
`.mod` deposit transcribes the **Cav3.1** constant set
(`F = 9.6485e4`, `R = 8.3145`, `kelvinfkt = 273.19 + celsius`). The
`_Frozen` siblings carry `ghk = _cav3p1_nmodl_ghk_flux` and reproduce the
mod's numbers; `_FrozenCav3p1Ghk` states this once and quantifies it as
"around 5e-4 relative".

Measured directly at fixed gates (`p = q = 0.6`, 36 degrees C,
Ci = 2.4e-4 mM, Co = 2 mM), `Cav2p1_MA2024_PC` against
`Cav2p1_MA2024_PC_Frozen`:

| V (mV) | unfrozen (mA/cm2) | frozen (mA/cm2) | relative |
| ---: | ---: | ---: | ---: |
| -80 | +1.104198e-01 | +1.104045e-01 | 1.39e-04 |
| -40 | +5.795022e-02 | +5.794331e-02 | 1.19e-04 |

and the free run of the `_Frozen` configs is clean on all three
templates:

| config | max abs dV | max abs dI | max gate err |
| --- | ---: | ---: | ---: |
| `cav2p1_ma24_pc_frozen` / vinit | 3.091e-10 | 2.839e-14 | 3.047e-12 |
| `cav2p1_ma24_pc_frozen` / dc | 1.856e-09 | 3.910e-12 | 3.111e-11 |
| `cav2p1_ma24_pc_frozen` / ac | 4.983e-10 | 7.260e-15 | 1.919e-12 |

`Cav2.1` therefore has no gating or kinetic defect: the residual is the
constant-set difference between the two registered variants, and it is
removed by pointing the comparison at the variant that carries the mod's
constants. That statement is about the current *value* only -- the same
`_Frozen` classes also change the solver's linearisation, which is the
subject of the next section. That is a config decision, not a channel change, and it should
be made deliberately -- the unfrozen class is what the cell models
instantiate.

### `Cav3.1`: the residual is not in the current law

The three `cav3p1` configs fail every template, with `dI` up to
`9.9e-05` and gate `p` up to `7.9e-04`. `cav3p1_ma24_pc_frozen` is clean
(`6.4e-10` / `2.7e-13` / `9.0e-12`), and `cav3p1_test_pc24`, which pairs a
deliberately simplified ohmic test mechanism with its own BrainCell class,
is also clean (`3.1e-10` / `5.8e-17`). The frozen and unfrozen classes are
however **numerically identical on the forward path**, which was verified
rather than assumed:

| check | unfrozen | frozen |
| --- | --- | --- |
| `ghk` helper | `_cav3p1_nmodl_ghk_flux` | `_cav3p1_nmodl_ghk_flux` |
| I at p=q=0.6, V=-50 mV (mA/cm2) | 8.009642e-02 | 8.009642e-02 |
| steady-state `p` at -49.945 mV | 0.6013245210505895 | 0.6013245210505895 |
| steady-state `q` | 0.041063508017586556 | 0.041063508017586556 |
| I at those gates (mA/cm2) | 5.5004583173319625e-03 | 5.5004583173319625e-03 |
| `freeze_gradient(V)` round-trip | exact | exact |

The residual enters on the first step of the free run, and it is caused by
`freeze_drive_gradient` itself, not by the channel. Flipping that one class
attribute is enough to swap the two results exactly, in either direction:

| run | max abs dV | max abs dI | max gate err |
| --- | ---: | ---: | ---: |
| `Cav3p1_MA2024_PC` as shipped (`freeze` off) | 1.804e-01 | 9.853e-05 | 7.940e-04 |
| `Cav3p1_MA2024_PC` with `freeze` on | 2.890e-09 | 1.407e-12 | 3.708e-11 |
| `Cav3p1_MA2024_PC_Frozen` as shipped (`freeze` on) | 2.890e-09 | 1.407e-12 | 3.708e-11 |
| `Cav3p1_MA2024_PC_Frozen` with `freeze` off | 1.804e-01 | 9.853e-05 | 7.940e-04 |

The reason is that the DHS/staggered solver does not use the current, it
uses its **derivative**. `Cell._voltage_linearizer` differentiates
`total_membrane_current / C` with `brainstate.transform.vector_grad`
(forward mode), and `lax.stop_gradient` -- which is what
`channel._base.freeze_gradient` wraps the drive in -- is transparent to
neither `jvp` nor `vjp`. So freezing the drive also zeroes the channel's
differential conductance:

| `dI/dV` of `Cav3p1_MA2024_PC` at -50 mV, p=q at steady state | value (mA/cm2 per mV) |
| --- | ---: |
| analytic, `freeze` off | -1.00074e-04 |
| analytic, `freeze` on | 0.0 |
| finite difference, step 0.001 mV | -1.00074e-04 |
| finite difference, step 0.1 mV | -1.00074e-04 |

The frozen variant therefore integrates the channel as a
voltage-independent source, which is a different forward model, and the
free run is sensitive to it: the case depolarises from `-49.9 mV` to
`-0.49 mV`, so a `1e-3`-relative difference in the first step's current
becomes `3.9e-02 mV` and, through the gates, `1.7e-04`. Both runs are
deterministic -- re-running the two configs reproduces every row bit for
bit, and re-running four configs of the survey reproduced them identically
-- so this is not measurement noise.

So the docstring claim that the frozen classes change "the gradient path,
not the forward numerics" holds only for `current()` itself; it fails
whenever anything differentiates the current, which the default solver
does. `_FrozenCav3p1Ghk` is worse: it sets `ghk` **and**
`freeze_drive_gradient`, so the three Cav2.1 `_Frozen` configs differ from
their siblings in the current value (~1.4e-04 relative, table above) *and*
in the linearisation. Any free-run comparison through them measures both.

### `CaLVA_SU2015_DCN` and `Na_ZH2019_IO`: spike-time amplification

Both are the same phenomenon: the failing case drives a **regenerative
excursion**, and the excursion amplifies the first step's `1e-9`
difference. The starting-point agreement is not the failing quantity --
what fails is the upstroke. After the event the two traces re-converge to
machine exactness, which is the actual statement about the channel.

| config | template | excursion of `v` | `dI` at first sample | peak `dI` | time of peak | `dI` at end |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `calva_su15_dcn` | vinit_celsius | -75.4 to +120.3 mV | +5.2e-10 | 2.127e-05 | 0.17 ms | -4.1e-14 |
| `calva_su15_dcn` | dc | -52.3 to +112.5 mV | +4.2e-08 | 1.648e-06 | 0.14 ms | +1.0e-10 |
| `na_zh19_io` | vinit_celsius | -50.0 to +50.0 mV | +3.4e-19 | 3.435e-07 | 1.32 ms | +4.4e-16 |

`calva`'s gate `m` starts `6.5e-11` from NEURON's value, peaks at
`6.8e-07` during the upstroke and returns to `1.1e-16`; its voltage
difference runs `9.8e-08` -> `3.2e-04` -> `8.2e-12` mV. `na_zh19_io` goes
`1.3e-09` -> `2.6e-05` -> exactly `0.0` mV. In both cases the channel is
right at the start and exactly right at the end; only the amplifier in the
middle fails the `1e-5` gate.

That means there is nothing to fix in `calva` or in `na_zh19_io`. Their
numbers can only be pushed under `1e-5` by removing the amplifier --
either by matching the first step to NEURON's integrator, or by comparing
under a voltage clamp instead of a free run.

## Assembled cells against NEURON

Each cell directory under `examples/neuron_compare/cell/` has a `run.ipynb`
that builds the whole cell twice -- once from the `.mod`/`.asc` deposits in
NEURON, once from `braincell` -- and runs both with the same clamp. Running
all of them needs a **full-cell** mechanism library: `nrnivmodl channel ion
synapse other` from `Cerebellum_mod/<CELL>/`, which lands in
`Cerebellum_mod/<CELL>/x86_64/` (gitignored). The `channel/x86_64` builds
alone are not enough: they omit the `Cdp*` calcium-pool mechanisms that
every assembled cell inserts.

All seven were run at the notebook's own settings (the model's `dt`, 20 to
100 ms, one somatic `IClamp`), `precision=64`, on CPU. Error is
`|V_braincell(t) - V_neuron(t)|`, with NEURON interpolated onto the
BrainCell sample times (`BrainCell` records after each step, NEURON also at
`t = 0`).

| cell | spikes n/bc | first-spike `dt` (ms) | max abs dV before first spike | max abs dV, whole trace | dRMS | max abs dV, last 50 samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `io_zh2019` | 3 / 3 | 5.3e-15 | 3.2e-09 | 9.8e-09 | 4.4e-10 | 1.7e-11 |
| `grc_ma2020` | 0 / 0 | -- | 3.1e-07 | 3.1e-07 | 1.5e-07 | 1.6e-08 |
| `pc_ma2024` | 2 / 2 | 1.5e-14 | 4.4e-05 | 7.0e-02 | 6.3e-03 | 6.9e-04 |
| `sc_ma2021` | 7 / 7 | 0.0 | 1.3e-04 | 6.8e-02 | 5.2e-03 | 3.4e-04 |
| `bc_ma2025` | 2 / 2 | 1.3e-13 | 5.3e-04 | 1.0e-02 | 6.6e-04 | 1.4e-05 |
| `goc_ma2020` | 7 / 7 | 4.6e-14 | 4.8e-04 | 5.1e-01 | 4.1e-02 | 3.2e-02 |
| `dcn_su2015` | blocked | -- | -- | -- | -- | -- |

`dcn_su2015` cannot run on this checkout: its morphology is read from
`/home/swl/Cerebellum_circuit/DCN/DCN/DCN_mor.hoc`, which is not present.

Reading the table:

- **Spike times agree exactly.** Every cell produces the same number of
  spikes, and the first spike lands on the same sample to within `1e-13`
  ms -- the sampling resolution. Assembly, mechanism insertion and the
  somatic geometry are therefore aligned.
- **Pre-spike subthreshold agreement spans five decades** across the
  catalogue: `3e-09` for `io_zh2019`, `3e-07` for `grc_ma2020`, and
  `4e-05` to `5e-04` for `pc`/`sc`/`bc`/`goc`. All are below the
  `1e-3 mV` whole-cell bar, so no assembled cell is misassembled, but the
  `1e-04` group is where any remaining per-channel work would show up.
- **After the first spike the comparison stops being a fidelity metric.**
  A spike is a regenerative event, so a `1e-04` subthreshold offset
  becomes `1e-02` to `5e-01` mV once the cells are past threshold. This is
  the same amplifier that dominates `Cav3.1`, `calva` and `na_zh19_io` in
  the single-channel sweep, and it is why the whole-trace column is two to
  four decades worse than the pre-spike column.


