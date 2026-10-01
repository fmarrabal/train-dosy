# TRAIn-MF signed-data correction and diagnostics (software 0.2.1)

[Español](TRAIN_MF_SIGNED_ES.md)

This revision uses installed MATLAB `lsqnonneg` and protocol **signed-v2.2**. The preceding signed-v2.1 port corrected attenuation-minimum subtraction and clipped component targets in historical TRAIn-MF and replaced the DiffAtOnce operational rank selector with held-out prediction. Version 0.2.1 additionally separates numerical convergence from compatibility with independently supplied noise, includes a null model, checks diffusion-grid boundaries and reports active rank. It does not recalibrate diffusion, translate peaks, or fit expected molecular masses.

`TRAIn-MF` is the new API/CLI identifier. `MF-AUTO` continues to call the frozen constrained solver used in the manuscript. `matlab/legacy/TRAIn_DOSY_MFV31.m`, the frozen solver, historical benchmarks and submitted manuscript are unchanged. These are successive implementations of the same multifrequency method, with different optimization/selection protocols, not interchangeable numerical results.

## Mathematical model

For phase-corrected signed observations Y (gradients by selected spectral frequencies), use K(i,l)=exp(-b(i)D(l)), with b in s/m² and D in m²/s. Minimize the data objective

    F(S,A) = (1/2) ||K S A - Y||_F²,
    S >= 0, sum(S(:,k)) = 1, A >= 0.

X=S A contains signal mass per diffusion node. A shared profile may be broad or multimodal; the factor count is not a species count. At least 256 increasing diffusion nodes are required. Division by log-bin width is needed to display density.

The algorithm uses one common RMS scale within each training split and restores output amplitude; validation observations do not influence candidate fitting scales or tolerances. It never subtracts an attenuation's minimum, rectifies negative observations, normalizes columns separately, smooths spectral amplitudes after optimization, or guesses b units. A nonzero final echo is real data, not automatically a baseline estimate. A physical baseline must be established upstream.

For each profile k, form the signed conditional least-squares target

    R_k = Y - sum(j != k) (K S(:,j)) A(j,:),
    y_k = R_k A(k,:)' / ||A(k,:)||².

TRAIn solves the physical-kernel problem in h=eta² using Gauss–Newton products and Steihaug trust-region steps. The accepted h is normalized to unit mass; all amplitudes are reoptimized with MATLAB NNLS. The proposal is retained only if the global data residual does not increase. No added lambdaS/lambdaA penalty is hidden in the kernel or in the NNLS residual reference.

TRAIn stops at `||K h-y_k|| <= term_factor * ||K h_NNLS-y_k||`, subject to a documented floating-point tolerance of `100*eps*||y_k||`. Default term_factor=1.05, allowed 1.02–1.05. This is iterative regularization using an **operational numerical reference**, not a discrepancy principle with independently measured noise. It is not exact block minimization and does not certify joint KKT stationarity or global optimality.

Compared with DiffAtOnce signed-v2, this port keeps small output weights, checks seed convergence, and allows 2000 inner iterations by default. Version signed-v2.2 also rescales each inner problem homogeneously and gives its initialization a relative positive floor: both zero and positive subnormal echoes previously could make the squared-variable initialization vanish. An NNLS zero solution is handled explicitly. These documented safeguards mean bitwise C#/MATLAB equivalence is not claimed.

## Automatic rank and status

Automatic selection includes **r=0** (the zero prediction) alongside ranks 1 through r_max; it requires at least eight gradients, including when r_max=1. Nonzero ranks are fitted without validation rows. Eligible predictions must be finite and have active factors. With known sigma, each training-derived prediction is tested against zero on the independent validation rows using a Gaussian projection score and a Bonferroni threshold across positive candidate ranks. If an eligible positive candidate supports signal, zero is removed before comparing the supported positive ranks. The smallest candidate within one paired standard error of the best is then refitted using all rows. Without sigma, zero competes directly in the one-standard-error rule. The rank comparison is a heuristic, not a confidence interval or a guarantee of globally optimal rank.

`candidate_numerical_converged` records numerical stopping independently of candidate eligibility. A finite candidate that stagnated can influence the predictive ranking: `rank_selection_resolved` only says the specified rule made a decision with the selected number of active factors after refitting. It does not assert optimizer convergence, a global optimum or chemical identification. Exactly inactive factors are removed from returned S/A; their nominal `selected_rank` remains visible and the result is unresolved when `active_rank` differs.

The diagnostics answer different questions:

| Field | Interpretation |
|---|---|
| `numerical_converged` | The applicable numerical residual/subproblem criteria were satisfied. This is not validation of D. |
| `model_compatible` | The residual passed the independent-noise check described below; NaN in MATLAB or null in JSON means unknown because sigma was not supplied. |
| `selected_rank`, `active_rank` | Selected factor count and number of active factors in the final fit. Neither counts chemical species. |
| `rank_selection_resolved` | The automatic prediction rule made a decision and the final active count agrees. Fixed rank sets `rank_selection_applicable=false` and `rank_selection_resolved=false`. |
| `boundary_hit`, `component_boundary_hit` | A column distribution or shared profile peaks at an endpoint or accumulates near a diffusion-grid boundary. Inspect and, when scientifically justified, rerun with an independently chosen wider grid. |
| `success` | All applicable numerical, model-compatibility, rank and boundary checks passed. This still does not establish physical uniqueness or chemical identity. |

For a nonzero fit, `success` requires numerical convergence, known model compatibility, resolved automatic rank and no column/profile boundary hit. Otherwise the result requires review; fixed-rank results therefore have `success=false` even if their numerical fit is good. Default boundary flags detect an endpoint mode or at least 10% of mass in the outer 2% at either end of the log-D interval (combined); these are adjustable warning heuristics, not a bias correction. A correct predictive rank and accurate D mean can coexist with `numerical_converged=false`, because independent NNLS fits have more freedom to fit noise. No extra factor is inserted merely to force this diagnostic to pass. A compatible null result can have `success=true` with `no_signal_supported`: it supports retaining the zero model under these tests, not an identified diffusion coefficient. Its X is zero, S has shape nD by 0, A has shape 0 by n_selected, and D mean/mode are undefined.

The optional direct-MATLAB option `sigma` is a **positive scalar standard deviation determined independently of the fitted residual**, in the same units as Y. The model check assumes iid Gaussian noise across all observed cells (including between frequencies), and compares squared residual/sigma² with a default 99% chi-square upper threshold using N supplied observations. This assumes independent errors, not independent underlying spectral signals: shared correlated signal profiles remain the purpose of MF. The conservative lack-of-fit screen does not estimate effective fitted degrees of freedom, supply a calibrated post-selection p-value, or prove identifiability. Correlated, heteroscedastic or baseline-contaminated errors violate the stated assumptions. Omitting sigma permits numerical computation but leaves `model_compatible` unknown and `success=false`; NNLS residuals cannot substitute for independent noise. Scale sigma by the same factor whenever Y is rescaled.

Direct MATLAB calls select about one quarter of non-endpoint acquisition rows deterministically, or accept a logical `validation_rows`. Construct the spectral signal mask without using those rows. The API explicitly uses the same reserved rows as its existing mask discovery. The API requires sigma and forwards it to both discovery and the independent model check; the inner numerical NNLS reference remains separate. Independent external gradients are required for unbiased evaluation.

## Run

```matlab
addpath('matlab');
D = logspace(log10(.02e-9),log10(5e-9),256)';
% sigmaIndependent comes from an independent, appropriate noise measurement.
[X,D,info] = TRAIn_DOSY_MF_Signed(Y,b,D, ...
    struct('r_max',4,'sigma',sigmaIndependent));
% Alternatively: historical grid units, with the revised algorithm:
[X,D_nano,info] = TRAIn_DOSY_MF(Yfull,b,[.02 5 256], ...
    struct('signal_mask',signalMask,'n_components','auto','r_max',4, ...
           'sigma',sigmaIndependent));
```

`signal_mask` is a logical vector selecting whole signal regions. Excluded columns in the compatibility wrapper are NaN (unestimated). `info.A` and the solver diagnostics refer to selected columns in `info.process_idx` order. `auto_r=false` requires an explicit numeric `n_components`; omitting it raises `TRAInMF:FixedRank` rather than silently performing automatic selection. Old settings that change preprocessing, smoothing, guessed units or penalties are rejected rather than silently ignored. The historical V3.1 remains available explicitly for replay.

With the licensed MATLAB backend configured as in [API documentation](API_EN.md):

```sh
train-dosy examples/one_component/input.json signed-result.json --method TRAIn-MF
```

Python and the C# SDK call this same MATLAB implementation; no independent C# solver is added by this release. The request can set `max_components` from 1 to 4 as a search cap. An HTTP 200 indicates completed computation, not successful numerical/chemical validation. The revised result has `protocol`, `diagnostics`, S and A; `kkt` is null because this algorithm does not calculate a joint KKT certificate.

## Reproduce the checks

```matlab
addpath('matlab/tests');
report = test_train_mf_signed('verification/train_mf_v22_recovery.json');
guards = test_train_mf_guards('verification/train_mf_v22_guards.json');
stress = test_train_mf_v22_stress('verification/train_mf_v22_stress.json');
```

The original fixed-seed suite remains intact: 1, 2 and 3 broad shared profiles, correlated spectral intensities and overlap, plus scaling, permutation, signed/zero tails, masks and minimum-subtraction bias. The guard suite adds independently known synthetic sigma, noise-only and zero inputs, true D below/above the supplied grid, absent/invalid sigma, fixed-rank option handling, and a positive 1e-320 echo. It reports convergence independently of accuracy and rejects false approvals; it does not force every exact or noiseless inverse problem to converge. Truth is used only by evaluation. Stored regression evidence is synthetic, not new experimental validation; good means/predictions do not guarantee correct modes or widths. The full-map limitations documented in DiffAtOnce are not declared solved.

The additional stress suite records 20 independent synthetic noise realizations, intensity factors 1e-150 and 1e150 with sigma scaled consistently, isolation of candidate fitting from changed validation values, and independent algebra/finite-difference checks of the conditional gradient and Gauss–Newton products. The stored signed-v2.2 run selected zero in all 20 null realizations; that finite sample does not calibrate an experimental false-positive rate. Versioned output names preserve the older verification records.
