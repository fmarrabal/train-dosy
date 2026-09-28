# TRAIn-MF signed-data correction (software 0.2.0)

[Español](TRAIN_MF_SIGNED_ES.md)

This revision ports the signed-data correction developed in DiffAtOnce to MATLAB, with installed MATLAB `lsqnonneg`. The protocol is **signed-v2.1**. It corrects attenuation-minimum subtraction and clipped component targets in historical TRAIn-MF; it does not recalibrate diffusion, translate peaks, or fit expected molecular masses. It also replaces the DiffAtOnce operational rank selector with held-out prediction.

`TRAIn-MF` is the new API/CLI identifier. `MF-AUTO` continues to call the frozen constrained solver used in the manuscript. `matlab/legacy/TRAIn_DOSY_MFV31.m`, the frozen solver, historical benchmarks and submitted manuscript are unchanged. These are successive implementations of the same multifrequency method, with different optimization/selection protocols, not interchangeable numerical results.

## Mathematical model

For phase-corrected signed observations Y (gradients by selected spectral frequencies), use K(i,l)=exp(-b(i)D(l)), with b in s/m² and D in m²/s. Minimize the data objective

    F(S,A) = (1/2) ||K S A - Y||_F²,
    S >= 0, sum(S(:,k)) = 1, A >= 0.

X=S A contains signal mass per diffusion node. A shared profile may be broad or multimodal; the factor count is not a species count. At least 256 increasing diffusion nodes are required. Division by log-bin width is needed to display density.

The algorithm scales all observations by one global RMS and restores the output amplitude. It never subtracts an attenuation's minimum, rectifies negative observations, normalizes columns separately, smooths spectral amplitudes after optimization, or guesses b units. A nonzero final echo is real data, not automatically a baseline estimate. A physical baseline must be established upstream.

For each profile k, form the signed conditional least-squares target

    R_k = Y - sum(j != k) (K S(:,j)) A(j,:),
    y_k = R_k A(k,:)' / ||A(k,:)||².

TRAIn solves the physical-kernel problem in h=eta² using Gauss–Newton products and Steihaug trust-region steps. The accepted h is normalized to unit mass; all amplitudes are reoptimized with MATLAB NNLS. The proposal is retained only if the global data residual does not increase. No added lambdaS/lambdaA penalty is hidden in the kernel or in the NNLS residual reference.

TRAIn stops at `||K h-y_k|| <= term_factor * ||K h_NNLS-y_k||`, subject to a documented floating-point tolerance of `100*eps*||y_k||`. Default term_factor=1.05, allowed 1.02–1.05. This is iterative regularization using an **operational numerical reference**, not a discrepancy principle with independently measured noise. It is not exact block minimization and does not certify joint KKT stationarity or global optimality.

Compared with DiffAtOnce signed-v2, this port keeps small output weights, handles exact-zero observations with a homogeneous nonzero initialization, checks seed convergence, and allows 2000 inner iterations by default. An NNLS zero solution is handled explicitly. These documented safeguards mean bitwise C#/MATLAB equivalence is not claimed.

## Automatic rank and status

Fit ranks 1 through r_max on acquisition rows excluding validation. Compare per-gradient prediction MSE; select the smallest eligible rank whose paired excess loss is no larger than one standard error relative to the best. Refit that rank on all supplied rows. Candidate eligibility requires successful TRAIn subproblems and active factors, not merely a smaller training residual. If all candidates fail those checks, retain a predictive candidate with unresolved status. This one-split rule is a heuristic, not a calibrated confidence interval.

The final NNLS target is checked **separately**: `train_mf_requires_review` is retained when the refit misses it, an inner solve fails, or a factor is inactive. A correct predictive rank can coexist with this status because independent NNLS fits have more freedom to fit noise. No extra factor is inserted merely to force this diagnostic to pass. Inspect `rank_selection_resolved`, `validation_mean_loss`, `rank_trials`, `success`, `residual_target` and `subproblem_failure_details`.

Direct MATLAB calls select about one quarter of non-endpoint acquisition rows deterministically, or accept a logical `validation_rows`. Construct the spectral signal mask without using those rows. The API explicitly uses the same reserved rows as its existing mask discovery. Sigma is used for API signal discovery; the revised solver's residual reference is not replaced by sigma. Independent external gradients are required for unbiased evaluation.

## Run

```matlab
addpath('matlab');
D = logspace(log10(.02e-9),log10(5e-9),256)';
[X,D,info] = TRAIn_DOSY_MF_Signed(Y,b,D,struct('r_max',4));
% Alternatively: historical grid units, with the revised algorithm:
[X,D_nano,info] = TRAIn_DOSY_MF(Yfull,b,[.02 5 256], ...
    struct('signal_mask',signalMask,'n_components','auto','r_max',4));
```

`signal_mask` is a logical vector selecting whole signal regions. Excluded columns in the compatibility wrapper are NaN (unestimated). `info.A` and the solver diagnostics refer to selected columns in `info.process_idx` order. Old settings that change preprocessing, smoothing, guessed units or penalties are rejected rather than silently ignored. The historical V3.1 remains available explicitly for replay.

With the licensed MATLAB backend configured as in [API documentation](API_EN.md):

```sh
train-dosy examples/one_component/input.json signed-result.json --method TRAIn-MF
```

Python and the C# SDK call this same MATLAB implementation; no independent C# solver is added by this release. The request can set `max_components` from 1 to 4 as a search cap. An HTTP 200 indicates completed computation, not successful numerical/chemical validation. The revised result has `protocol`, `diagnostics`, S and A; `kkt` is null because this algorithm does not calculate a joint KKT certificate.

## Reproduce the checks

```matlab
addpath('matlab/tests');
report = test_train_mf_signed('verification/train_mf_signed_regression.json');
```

The fixed-seed cases contain 1, 2 and 3 broad shared profiles, correlated spectral intensities and overlap. Truth is used only by evaluation. Tests cover mean D, external clean prediction, rank selection, scaling, frequency permutation, signed/zero tails, insufficient rank, masks and incompatible options. A separate surviving-tail case reproduces the D bias from minimum subtraction. Stored regression evidence is synthetic, not new experimental validation; good means/predictions do not guarantee correct modes or widths. The full-map limitations documented in DiffAtOnce are not declared solved.
