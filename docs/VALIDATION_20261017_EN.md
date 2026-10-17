# Validation supplement: 17 October 2026

This supplement records what the additional experiments showed, including failures and cases where a reference method performed better. It does not change the software 0.2.1 API or replace its signed-v2.2 TRAIn-MF backend. The original MF-AUTO results remain archived. Dates in folder names identify the study snapshot, not a journal publication or a new software version.

## Results and their limits

| Study | Independent inputs and design | What the results support |
|---|---|---|
| Noise-only selection | 300 matrices, 100 at each of three noise levels; paired RAI-S/DOME-S fits | A substantially larger null check than the original six controls. Report the noise strata separately; these observations are not a general false-discovery guarantee. |
| Weak components | 96 matrices: four relative amplitudes, three SNRs and eight repeats | The weak rate was recovered in 45/96 RAI-S and 44/96 DOME-S fits. Coherence screening does not make all weak components detectable. |
| Algorithm ablation | 36 signal matrices, paired screening/order-selection combinations | The shared screening and selection steps matter. Similar RAI-S/DOME-S outcomes do not demonstrate two independent improvements. |
| Regularized TRAIn-MF convergence | Nine rank/SNR matrices, each with four warm-iteration caps and two starts | 14/36 selected fits reached relative KKT <= 1e-6; 22 remained unresolved within the common budget. Rank is supplied in this controlled experiment. It does not validate automatic rank selection or establish that eight sweeps suffice. |
| Neural policy | 120 new test problems; five training seeds and one classical policy, 720 fits | On 105 signal problems the mean transport-error change was -0.00054 (paired interval [-0.00174, 0.00056]); the clean prediction-error change across all 120 was -0.00012 ([-0.00044, 0.00010]). Neither interval establishes a predictive advantage. All final fits passed the specified numerical checks. |
| Classical references | 47 analysed independent-noise matrices and 24 stress cases (12 shared truths under two distortions), seven methods | The archive contains 504 fits; 497 enter summaries after the same pilot matrix is excluded for all seven methods. SCORE-CV, SCORE-NNLS-CV, LOCODOSY, positive Tikhonov and the PALMA objective are compared under stated wrappers. PALMA did not meet its all-column convergence certificate within the budget, so its outcome is a finite-budget result. |
| Industrial resins | Four series, three diffusion times each, five methods: 60 fits | These are 12 blocks clustered within four series, not 12 independent mixtures. RAI-S/DOME-S reached the four-component cap and retained model mismatch. Only block-level and aggregate diagnostics are public. There are no independent chemical component counts or diffusion truths. |
| Public DOSY Toolbox example | 30 acquisitions, 4074 selected channels, seven external holdouts | Relative prediction RMSE was 0.582% for RAI-S/DOME-S, 0.232% for signed SCORE-CV, 0.582% for SCORE-NNLS-CV and 0.270% for local signed LOCODOSY. The converged signed SCORE fit was better on this example. Signed amplitudes are not a positive diffusion measure; no independent diffusion truth is supplied. |

The neural intervals are paired, descriptive bootstrap intervals. Seeds and repeated method fits do not multiply the number of independent test problems. Likewise, repeated resin diffusion times are not independent samples. All reported failure flags remain in the recorded outcomes.

The monitored MF entry point is `validation/mf_convergence/instrumented/mf_robust/DOSY_MF_Runtime.m`, relative to the dated study folder. It preserves the regularized reference 5.1 objective and adds runtime checks of the specified conditional curvature condition. This condition concerns a block of the optimization problem; it is not chemical identifiability. The archived illustration used reference 5.0, and neither of these references is relabelled as the signed-v2.2 API backend.

## Files and reproduction

The original API and benchmark commands remain in the main README. The new materials are under `paper/revision_complete_20261017/validation`, except the first null/ablation/distortion study, which retains `paper/revision_20261017/validation` because its imports depend on that location. Each study has its own protocol and summary. The manifest `PUBLIC_VALIDATION_MANIFEST.json` identifies the publicly supplied files and their hashes.

The public copy contains numerical sources, protocols, compact JSON/CSV outcomes, five neural checkpoints and their training histories, the frozen synthetic training traces, and the nine small synthetic MF input matrices. It deliberately omits large fitted arrays, neural per-case arrays, industrial observations and private laboratory or editorial documents. Recorded summaries can be inspected directly. A full prediction-level replay requires a fresh computation; a compact JSON result is not a substitute for an omitted array.

Use a separate environment for each recorded Python stack. The atomic/experimental environment records Python 3.13.13, NumPy 2.5.1 and SciPy 1.18.0. The neural study records Python 3.13.13, NumPy 2.5.3, SciPy 1.18.1 and CPU PyTorch 2.14.0. The complete pins are `requirements_revision_atomic.txt` and `requirements_revision_neural.txt`. MATLAB experiments used R2026a with Optimization Toolbox. A GPU is unnecessary. Operating-system and library changes can affect floating-point results and timings.

The external benchmark imports a sibling `diffusion-inversion` checkout. From the directory containing `train-dosy`, run:

```sh
git clone https://github.com/fmarrabal/diffusion-inversion.git
git -C diffusion-inversion checkout 2aecc92fd74f809f96f05d36f9da0f3a5ca512bd
```

Keep the two repositories side by side. This pin preserves the attributed SCORE/LOCODOSY translation and numerical dependencies used by the benchmark. The NNLS-amplitude SCORE variant is labelled separately from the signed-amplitude reference. PALMA here denotes the stated objective, implementation and cross-validation wrapper; it is not an assertion of byte-identical upstream software.

For summary-only replay, use a disposable copy of the published tree; aggregators write their output files. The following commands perform no inverse fits:

```sh
python paper/revision_20261017/validation/summarize_revision_validation.py
python paper/revision_complete_20261017/validation/check_compact_summaries.py
python paper/revision_complete_20261017/validation/external/analyse_external.py
```

For fresh atomic fits, make another working copy with the same repository-relative layout, copy the study scripts and dependencies, and leave that study's generated `results`/`inputs` directories absent. Run `run_frozen_additional_validation.py`, `run_frozen_distortion_stress.py`, `run_weak_power.py`, or `run_external_benchmark.py` as appropriate. Run `run_score_nnls.py` to add the explicit nonnegative-amplitude SCORE comparison. Do not reuse an old result directory: several runners resume completed cases by filename. These are local research computations, not CI jobs.

For an independent neural repeat, create an empty destination with the supplied path-only adapter:

```sh
python paper/revision_complete_20261017/validation/neural_ablation/prepare_portable_replay.py --destination neural-repeat
```

The adapter copies the numerical snapshot without changing seeds, settings or inference rules. `neural-repeat/replay_adaptation.json` lists the subsequent training, checking, inference and analysis commands. Install its neural requirements in a separate environment. The frozen five policies can also be inspected without training. The factorial study uses `factorial_study.py` at its original repository-relative location.

For MF, use a fresh directory containing the `.m`/`.py` drivers and the `source_frozen` and `instrumented` folders, without generated configuration or results. In MATLAB run `prepare_mf_experiment`, `test_runtime_curvature`, `run_mf_batch([1 2 3])` and `audit_completed_outputs`; then run `aggregate_mf.py`. See `EVIDENCE_EN.txt` for the exact objective, tolerances and replay checks. The compact public results include JSON summaries; the MAT prediction arrays needed by the completed-output audit are generated by the fresh run. The existing source hashes identify the original run, so a new directory has a separate provenance record.

## Experimental scope and third-party material

The resin runner is published for inspection, but its observations are not distributed. It cannot reproduce the industrial fits from this public subset alone. `summary_public.json` and `metrics.csv` retain their numerical values and diagnostic flags; private material-identity crosswalks are excluded.

The public Varian example is retrieved from the Manchester NMR Methodology Group's official DOSY Toolbox archive. Its exact URL and SHA-256 are in `experimental/public_mixture/toolbox_legacy_download.json`. From that folder, in an atomic-study environment, run:

```sh
python download_varian.py
python prepare_varian.py
python run_varian.py
python summarize_varian.py
```

Download must precede the raw-file hash audit. Preparation uses the frozen split and a common phase estimated only on discovery acquisitions. It retains all channels within the training-detected positive absorption support; channels outside it are unestimated, not evidence of absence. Neither phase nor kernel scaling uses held-out responses or a fitted chemical identity. This example supplies experimental predictive evidence, not independently measured diffusion coefficients. The public source has no explicit data licence in the recovered legacy archive, so the repository supplies its downloader and attribution rather than redistributing the measured arrays.

The software is GPL-3.0-or-later except where a retained third-party notice says otherwise. Preserve `NOTICE`, the GNAT GPL notice, the SCORE/LOCODOSY attribution, and the BSD nmrglue licence. The pinned `diffusion-inversion` dependency has its own `THIRD_PARTY_NOTICES.md`, `licenses/` and `docs/LOCODOSY_SOURCE_NOTICE.md`. MATLAB and its `lsqnonneg` implementation are separately licensed and are not redistributed. No GitHub Actions workflow is added or enabled.

`EXCLUDED_SCIENTIFIC_ARTIFACTS.json` lists the omitted synthetic and public-example artifacts by path, size and SHA-256. It records local evidence, not downloadable release assets. Private industrial observations are outside this inventory as well as outside the public package. All 80 synthetic training traces, all nine MF input matrices and all five neural checkpoints are included.
