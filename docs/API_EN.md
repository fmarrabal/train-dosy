# API contract (v1)

The public algorithm name is **TRAIn-MF**. Software 0.2.1 uses protocol `signed-v2.2` for identifier `TRAIn-MF`; `MF-AUTO` preserves the frozen constrained manuscript solver (`DOSY_MF_Auto`). See the [correction and migration guide](TRAIN_MF_SIGNED_EN.md). These are explicitly different numerical protocols within the same method's development.
The service performs joint inversion over selected frequencies. It does not receive a chemical identity, true rank, target diffusion value, or truth file.

## Run locally
From the repository root, install the Python API extra and run:
~~~sh
python -m uvicorn train_dosy.api:app --host 127.0.0.1 --port 8765
~~~
Interactive documentation: http://127.0.0.1:8765/docs . Machine-readable schema: /openapi.json . Health: GET /health . Inversion: POST /v1/fit with application/json.

~~~sh
curl -X POST http://127.0.0.1:8765/v1/fit -H "Content-Type: application/json" --data-binary @examples/three_components/input.json -o result.json
~~~
On Windows PowerShell use curl.exe to avoid the legacy alias.

## Request fields
| Field | Meaning |
|---|---|
| Y | Real signed phased data, rows = gradients, columns = acquired chemical shifts. Negative noise is retained. |
| b | One distinct nonnegative value per row, in s/m². This is the physical attenuation factor, not raw gradient squared. |
| ppm | Distinct chemical shifts, matching the columns, in either order. |
| sigma | Required positive scalar noise standard deviation measured independently of the fitted residual, in the same signal units as Y. The TRAIn-MF model check assumes homogeneous iid Gaussian noise. |
| method | RAI-S (default), DOME-S, or MF-AUTO / TRAIn-MF with a configured licensed MATLAB backend. |
| mask | Optional Boolean frequency mask. True means include. Excluded frequencies are not estimated. |
| diffusion_bounds | Positive increasing pair, in m²/s. Default [1e-10, 1.5e-8]. |
| bins | 256 (default) through 2048. For atomic methods this affects the exported display grid, not off-grid rate optimization. TRAIn-MF actually optimizes masses on this grid. |
| max_components | Search cap 1–4; default 4. It is a limit, not a supplied rank. Only frozen MF-AUTO requires the fixed cap of 4. |

The contract rejects extra fields, nonfinite inputs, repeated b or ppm coordinates, wrong shapes, empty masks, fewer than 12 or more than 256 acquisition rows, more than 8192 columns, and more than 524288 observations. The HTTP body limit is 16 MiB. Split-independent signal masking happens before inversion; there is no processing of all 65k baseline points.

When mask is omitted, discovery uses the first six available low-b rows excluding the internal validation rows, thresholds their mean at 4 sigma/sqrt(n_discovery), and dilates by two acquired-frequency positions. Supply a carefully designed mask for solvent exclusion, nonuniform axes, phase artifacts, or very broad features. Input column adjacency is assumed for dilation. Screening is not proof that excluded signal is absent.

An explicit mask must be predetermined or derived without external test data and preferably without internal validation data. The historical benchmark uses its frozen outer-training mask for exact reproducibility; this includes internal validation information. Its selector is therefore reported as a heuristic, not an independently calibrated test.

## Response
Atomic output D contains off-grid diffusion coefficients; A has shape r by n_selected. X has shape bins by n_selected and stores **mass per bin**, never density. logD_edges gives natural-log SI diffusion cell boundaries; divide X by the corresponding log width to obtain a density. prediction has input-row order and selected-frequency columns and uses exact off-grid rates. It is not recomputed from rounded display bins.

Both MF protocols output D_grid, X, S and A with X=S A; their predictive factor count is not the number of molecular species. MF-AUTO reports a KKT residual. TRAIn-MF reports protocol `signed-v2.2`; its KKT field is null because it does not calculate joint KKT stationarity. Its diagnostics separate `numerical_converged`, `model_compatible`, `selected_rank`, `active_rank`, `rank_selection_resolved`, `boundary_hit` and `component_boundary_hit`. Exactly inactive factors are removed from S/A, whose factor dimension is `active_rank`; nominal `selected_rank` is retained and an activity mismatch leaves the result unresolved. `candidate_numerical_converged` records candidate stopping separately: the predictive selection can use finite active candidates that did not attain their numerical target. `success` requires all final checks to pass. Numerical convergence alone cannot approve a biased or incompatible diffusion estimate.

For TRAIn-MF, required API sigma is forwarded to signal discovery, held-out evidence against the null, and a separate residual-compatibility check, with a default 99% chi-square upper threshold for N observations under iid Gaussian errors across all cells. Correlated underlying spectral signals are allowed; this assumption concerns measurement noise. Sigma does not replace the numerical NNLS reference. The model check is a conservative lack-of-fit screen, not a calibrated post-selection p-value or proof of identifiability. Direct MATLAB also permits omitting sigma: then `model_compatible` is NaN (JSON null), and `success=false`. Unknown compatibility must never be coerced to true. Direct fixed-rank MATLAB results have `rank_selection_applicable=false`, `rank_selection_resolved=false` and `success=false`. Scale sigma together with Y when changing intensity units.

Both return mask, selected_frequency_indices (zero-based), ppm, selected_rank, selection_mode, search_limit, search_limit_reached, units and software_version. TRAIn-MF includes the null model r=0 in automatic selection. A supported zero rank has X=0, S of shape bins by 0, A of shape 0 by n_selected, zero prediction, and undefined D mean/mode. JSON represents S as a list of empty rows and A as []; reconstruct A's second dimension from the number of selected frequencies. A `no_signal_supported` status may have `success=true`, meaning that the null model passed the stated checks; it does not identify a component. Automatic discovery with no selected frequencies returns 422 instead of claiming a zero-signal measurement.

Atomic diagnostics include candidate losses, support decisions, stationarity, boundary hits, poor rate precision and atomic-model mismatch. The conditional score and local pseudoinverse uncertainty are not calibrated chemical confidence intervals. Nonfinite diagnostic values serialize as null; arrays with physical inputs never accept nonfinite values.

## Status and deployment
200 = completed computation; inspect success. 422 = invalid data/no detected signal. 429 = another fit is running. 503 = no verified candidate or unavailable MATLAB backend. Unexpected failures are 500 with no stack trace in the HTTP response. One numerical request runs at a time. No remote URLs, shell commands or server file paths are accepted in requests. Browser cross-origin access is not enabled.

This is a locally tested research API, not a hosted service. An internet deployment needs authentication, a reverse proxy with time limits, resource isolation and an operational review. Python solver jobs have no wall-clock cancellation; stopping an HTTP client is not guaranteed to interrupt computation. MATLAB subprocesses have a 600-second limit.

## Optional native TRAIn-MF backend
Install MATLAB with Optimization Toolbox and set these environment variables in the server process:
~~~powershell
$env:TRAIN_DOSY_MATLAB = 'C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$env:TRAIN_DOSY_MATLAB_FUNCTIONS = (Resolve-Path matlab).Path
~~~
Then start the server or use the CLI with `--method TRAIn-MF` (revised) or `--method MF-AUTO` (frozen). Only the operator chooses the executable and MATLAB code path. Each request starts a separate MATLAB batch process. MATLAB is neither bundled nor redistributed. Direct revised MATLAB usage and reproducible checks are in [the correction guide](TRAIN_MF_SIGNED_EN.md); `matlab/example_mf.m` retains the manuscript example.
