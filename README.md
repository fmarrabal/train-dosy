# TRAIn-DOSY: positive joint Laplace inversion
[Español](README_ES.md) · [API](docs/API_EN.md) · [Mathematics](docs/METHODS_EN.md) · [Provenance](docs/PROVENANCE.md) · [Manuscript](paper/output/pdf/Positive_Joint_Laplace_Inversion_Mathematics.pdf)

Research software by **Francisco M. Arrabal-Campos** for joint DOSY reconstruction from whole spectral signal regions. The TRAIn-MF line derives from the original **TRAIn of Xu and Zhang**, through Arrabal-Campos's **TRAIn_DOSY_MFV31**. Both the preserved historical source and the later numerical TRAIn-MF reference are included and explicitly distinguished.

![DOSY examples with ground truth](examples/dosy_ground_truth.png)

Current manuscript: [manuscript-v6](https://github.com/fmarrabal/train-dosy/releases/tag/manuscript-v6), 38 pages and 48 references. This revision names Codex as a programming assistant for the repository and Trinka as a language-editing assistant for the manuscript, without model or version details. It retains the author metadata, funding and mathematical audit introduced in v5. Numerical software v0.1.0 and its results are preserved. See the [manuscript audit](paper/AUDITORIA_MANUSCRITO_ES.md).

## Why TRAIn-MF?
**MF means multifrequency** here, consistent with the original `TRAIn_DOSY_MFV31` header. The mathematical implementation is a nonnegative matrix factorization, `Y ≈ K S A`: columns of `S` are shared diffusion profiles and rows of `A` describe their intensities across chemical shifts. It still processes multiple frequencies when the selected rank is one. A profile can be broad or multimodal; rank is not a chemical species count.

**TRAIn-MF: a multifrequency extension of TRAIn through nonnegative matrix factorization** is the algorithm name. The current constrained reference replaces the original embedded TRAIn iteration; it is not numerically identical to V3.1. Function names, historical source bytes, numerical cores and frozen results are preserved. See the [expanded bibliography](docs/BIBLIOGRAPHY_EN.md).

## What is implemented
| Language | Implementation | Role |
|---|---|---|
| MATLAB | Historical TRAIn_DOSY_MFV31; native DOSY_MF_Auto and constrained TRAIn-MF reference | Distributional/polymer profiles, ≥256 diffusion bins, installed MATLAB NNLS |
| Python | Frozen RAI-S and DOME-S solvers, validation, CLI, REST API, synthetic benchmark and plotting | Discrete shared diffusion rates and automatic spectral-support/order selection |
| C#/.NET 10 | Typed SDK and command-line client | Calls the same API; not a third independent numerical solver |
| API TRAIn-MF backend | Python launches the native MATLAB code when explicitly configured by the operator | Preserves the MATLAB algorithm and licensing requirements |

The numerical TRAIn-MF reference and frozen atomic core are copied without alteration and hashed. The public packaging adds no manual peak-height deletion. The algorithm receives neither ground truth nor an expected component count.

## Install Python
Python 3.11 or newer; the release was tested with Python 3.13.13. Use a virtual environment:
~~~sh
git clone https://github.com/fmarrabal/train-dosy.git
cd train-dosy
python -m venv .venv
~~~
Windows:
~~~powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[api,test,plots]"
~~~
Linux/macOS:
~~~sh
source .venv/bin/activate
python -m pip install -e ".[api,test,plots]"
~~~
requirements-lock.txt records the versions used for the release. For that environment, install it before the editable package; cross-platform bitwise equality is not promised.

## Run a complete example
~~~sh
train-dosy examples/three_components/input.json result.json --method RAI-S
train-dosy examples/three_components/input.json dome-result.json --method DOME-S
python scripts/make_examples.py
python scripts/plot_examples.py
~~~
The first two commands fit measured-style inputs. The last two regenerate fixed one-, two- and three-component examples and their DOSY comparison against truth. input.json contains only observations and allowed settings. truth.json is separate and is never read by fit(). The saved results and examples/summary.json record numerical diagnostics and prediction error on eight external gradients.

Direct Python:
~~~python
import json
from train_dosy import fit
request = json.load(open("examples/two_components/input.json"))
result = fit(request)
print(result["selected_rank"], result["D"], result["success"])
~~~

## Input preparation
Supply a real, phase-corrected Y matrix: acquisition rows by chemical-shift columns. Retain signed noise. b is in s/m²; D is in m²/s and bD must be dimensionless. Do not substitute raw G² or guess pulse-sequence scaling. Correct phase/baseline and calculate physical b values upstream. Supply sigma in the same amplitude units as Y; this release assumes homogeneous known Gaussian noise.

Use the full spectrum to choose signal regions, then jointly fit those columns. A Boolean mask excludes solvent and noise-only regions; excluded values remain unestimated. Default discovery avoids the inner validation rows. Do not run separate one-pixel fits and then call the result joint component recovery. The array/resource limits are specified in the [API contract](docs/API_EN.md).

For RAI-S/DOME-S, a 256-bin map conserves estimated mass but the rate optimization is continuous. Increasing display bins does not improve physical resolution. For TRAIn-MF the grid contains the actual unknown masses. Display density requires division by bin width.

## MATLAB
MATLAB plus Optimization Toolbox is required; tested in R2026a. From the MATLAB prompt:
~~~matlab
addpath('matlab');
r = example_mf();
imagesc(r.ppm, log10(r.D), r.X);
set(gca,'XDir','reverse'); xlabel('ppm'); ylabel('log_{10} D (m^2/s)');
~~~
For direct use:
~~~matlab
addpath('matlab/reference');
D = logspace(-10, log10(15e-9), 256)';
% Y contains signal frequencies; b and sigma are in physical SI/signal units.
% mask discovery must not use validationRows or external test rows.
[X,D,info] = DOSY_MF_Auto(Y,b,D,sigma,innerRows,validationRows,struct());
~~~
innerRows and validationRows are disjoint logical vectors. The final refit uses their union. Always inspect info.status and info.fit.kkt when available; unresolved rank/convergence is not chemical identification.

matlab/legacy/TRAIn_DOSY_MFV31.m preserves the author's original V3.1 interface and defaults for provenance. Its D_params convention, optional b scaling, smoothing and thresholding differ from the later reference. The historical file is not the validated automatic selector used in this paper.

## REST API and C#
~~~sh
python -m uvicorn train_dosy.api:app --host 127.0.0.1 --port 8765
dotnet build csharp/TrainDosy.Cli -c Release
dotnet run --project csharp/TrainDosy.Cli -c Release -- examples/three_components/input.json csharp-result.json
~~~
Swagger: http://127.0.0.1:8765/docs . [Full endpoints, units, errors, payload limits and optional MATLAB backend](docs/API_EN.md). The C# CLI supports an optional third argument for the server URL and cancellation with Ctrl+C. A failed HTTP request returns a nonzero exit code; an unresolved solver result returns code 3.

Typed SDK:
~~~csharp
using var http = new HttpClient {
    BaseAddress = new Uri("http://127.0.0.1:8765/"),
    Timeout = TimeSpan.FromMinutes(11)
};
var result = await new TrainDosy.DosyClient(http).FitAsync(request, cancellationToken);
~~~
From MATLAB, request = jsondecode(fileread('examples/one_component/input.json')); result = dosy_api(request); calls the same local server.

## Benchmark and paper reproduction
~~~sh
python -m pytest -q
python benchmarks/run_atomic.py --count 48 --output benchmark_run
python paper/scripts/prepare_evidence.py
~~~
Compile on Windows with PowerShell:
~~~powershell
.\paper\build.ps1
~~~
On another platform, run pdflatex three times on main.tex from paper/ (MiKTeX/TeX Live and the required packages must be installed). The editable source and original MDPI template notices are supplied.

The atomic protocol contains 36 signal-present discrete cases, six null cases and six broad-profile controls. The frozen primary table reports correct order/all rates as RAI 17/36, RAI-S 23/36, DOME 22/36, DOME-S 23/36 (all-rates column). Selected order alone differs: 17, 25, 26 and 25 of 36 respectively. Agreement of the two S variants does not create independent replication. Broad polymer distributions require a different representation.

The original TRAIn-MF manuscript illustration is an unchanged saved result. The package does not claim new superiority, global optimality, chemical identification, new experimental validation or full retraining of the exploratory neural models. See [methods and reproducibility scopes](docs/METHODS_EN.md). Full-benchmark reruns may be slow; the three examples are smoke demonstrations, not replacements for the test panel.

## Layout
- src/train_dosy: contract, API, CLI and frozen Python estimator.
- matlab/reference: native TRAIn-MF, NNLS/simplex/profile optimization and automatic selection.
- matlab/legacy: original TRAIn_DOSY_MFV31.
- csharp: reusable SDK and CLI.
- examples: inputs, separate truth, fitted outputs and DOSY figures.
- benchmarks: frozen generator, evaluation and historical per-problem metrics.
- paper: source, PDF, minimal synthetic replay inputs, tables and figures.
- provenance and verification: source hashes and local validation evidence.
- docs: bilingual API/mathematical documentation and attribution.

## Citation, license and status
Use [CITATION.cff](CITATION.cff) for software and cite [original TRAIn](https://doi.org/10.1021/ac402698h) when discussing its lineage. Relevant author publications include [diffGA](https://doi.org/10.1039/c7sm01569k), [dART](https://doi.org/10.1021/acs.jpca.8b08584), and [regularized Kaczmarz](https://doi.org/10.3390/math13132166); the manuscript adds the molecular-weight studies and the 2016 correction.

Software: GPL-3.0-or-later; [NOTICE](NOTICE) preserves attribution and dependency distinctions. MATLAB is separately licensed. Release v0.1.0 is research software with a draft manuscript: author order, affiliations, ORCIDs, correspondence and funding were supplied by the corresponding author. CRediT roles are explicitly proposed for confirmation by all three authors; conflict declarations remain pending. No journal submission is implied. Validation runs locally; no GitHub Actions workflows are installed or required.

