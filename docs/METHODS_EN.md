# Mathematical contract and limits

## Why MF?
**MF means multifrequency** here, consistent with the original `TRAIn_DOSY_MFV31` header. The mathematical implementation is a nonnegative matrix factorization, `Y ≈ K S A`: columns of `S` are shared diffusion profiles and rows of `A` describe their intensities across chemical shifts. It still processes multiple frequencies when the selected rank is one. A profile can be broad or multimodal; rank is not a chemical species count.

**TRAIn-MF: a multifrequency extension of TRAIn through nonnegative matrix factorization** names the lineage. The current constrained reference replaces the original embedded TRAIn iteration; it is not numerically identical to V3.1. Function names, historical source bytes, numerical cores and frozen results are preserved. See the [expanded bibliography](BIBLIOGRAPHY_EN.md).

The acquisition model is Y(i,l)=integral exp(-b_i D) d mu_l(D)+noise. D is diffusion; inferred molecular weight requires a separately justified calibration.

## Distributional MF
Y≈K S A; K(i,j)=exp(-b_i D_j). S≥0, each column sums to one, A≥0. The reference minimizes
F = ||K S A - Y/s||_F²/(2p) + lambda_S ||L S||_F²/2 + lambda_A ||A||_F²/(2p).
Here s is the RMS of Y and L acts on log-diffusion density with quadrature. Returned A is rescaled to signal units. Default lambda_S=lambda_A=1e-8 in DOSY_MF_Auto; 256 bins, two starts and KKT tolerance 1e-6. Scalar decrease does not imply unique or globally optimal factors. The automatic selector uses noise-resolved singular directions as a search restriction and held-out predictive error. That rank is not an upper bound on chemical species. Inadequate stationarity or an insufficient rank cap is explicitly unresolved.

## Discrete RAI-S and DOME-S
Y≈exp(-b D) A, with positive continuous rates and nonnegative amplitudes. Candidate proposal paths differ. Amplitudes are solved by exact active-subset NNLS for at most four components; the reduced nonlinear fit uses the full active-set variable-projection derivative. Residualized kernel directions generate a score aggregated within separated spectral windows. Screened and unrestricted candidates compete on held-out gradients; the selector prefers fewer rates, then fewer support parameters within a paired prediction allowance. It never applies a fixed percentage of the tallest peak.

Defaults: max_components=4, alpha=.01, validation_stride=4, selection_se=1, max_nfev=350. Final amplitude KKT and scaled rate stationarity are checked. Local precision and model-mismatch flags must be reported. Scores with estimated rates and selected masks are heuristic; no familywise chemical-detection guarantee is claimed.

## Identifiability
Finite Laplace observations do not uniquely identify arbitrary positive measures. Narrow close components, weak species, proportional spectra, baseline/phase errors, noise misspecification and overly narrow bounds can defeat recovery. A fine grid cannot manufacture information. Recovering the fitted rank is weaker than recovering all diffusion rates and weaker still than chemical identification. The manuscript reports these separately.

## Reproduction scopes
- examples: three fixed development examples; regenerate with scripts/make_examples.py.
- benchmarks/run_atomic.py: exact historical 48-case test generator and four atomic methods; 36 signal-present discrete problems, six nulls, six broad misspecification controls. Its historical mask uses all outer training rows, including internal validation rows.
- benchmarks/frozen/metrics.csv: full existing per-case metrics including archived and exploratory neural results. No frozen numerical results were altered for the public release.
- paper/scripts/prepare_evidence.py: regenerate manuscript tables and figures from a minimal saved synthetic fixture set, with no solver calls.
- MATLAB example and optional API backend: actual native MF fit using licensed MATLAB. The manuscript's original MF figure remains the hashed saved output.
- C#: typed SDK/CLI for the same API. It is not an independent numerical reimplementation and provides no third independent scientific validation.
Neural training and the full historical multi-algorithm survey are outside this curated release. Their frozen reported metrics are retained; the release does not claim end-to-end retraining of those exploratory models.

