# Mathematical contract and limits

## Why TRAIn-MF?
**MF means multifrequency** here, consistent with the original `TRAIn_DOSY_MFV31` header. The mathematical implementation is a nonnegative matrix factorization, `Y ≈ K S A`: columns of `S` are shared diffusion profiles and rows of `A` describe their intensities across chemical shifts. It still processes multiple frequencies when the selected rank is one. A profile can be broad or multimodal; rank is not a chemical species count.

**TRAIn-MF: a multifrequency extension of TRAIn through nonnegative matrix factorization** is the algorithm name. The current constrained reference replaces the original embedded TRAIn iteration; it is not numerically identical to V3.1. Function names, historical source bytes, numerical cores and frozen results are preserved. See the [expanded bibliography](BIBLIOGRAPHY_EN.md).

The acquisition model is Y(i,l)=integral exp(-b_i D) d mu_l(D)+noise. D is diffusion; inferred molecular weight requires a separately justified calibration.

## Distributional TRAIn-MF
Y≈K S A; K(i,j)=exp(-b_i D_j). S≥0, each column sums to one, A≥0. The reference minimizes
F = ||K S A - Y/s||_F²/(2p) + lambda_S ||L S||_F²/2 + lambda_A ||A||_F²/(2p).
Here s is the RMS of Y and L acts on log-diffusion density with quadrature. Returned A is rescaled to signal units. Default lambda_S=lambda_A=1e-8 in DOSY_MF_Auto; 256 bins, two starts and KKT tolerance 1e-6. Scalar decrease does not imply unique or globally optimal factors. The automatic selector uses noise-resolved singular directions as a search restriction and held-out predictive error. That rank is not an upper bound on chemical species. Inadequate stationarity or an insufficient rank cap is explicitly unresolved.

## Discrete RAI-S and DOME-S
Y≈exp(-b D) A, with positive continuous rates and nonnegative amplitudes. Candidate proposal paths differ. Amplitudes are solved by exact active-subset NNLS for at most four components; the reduced nonlinear fit uses the full active-set variable-projection derivative. Residualized kernel directions generate a score aggregated within separated spectral windows. Screened and unrestricted candidates compete on held-out gradients; the selector prefers fewer rates, then fewer support parameters within a paired prediction allowance. It never applies a fixed percentage of the tallest peak.

Defaults: max_components=4, alpha=.01, validation_stride=4, selection_se=1, max_nfev=350. Final amplitude KKT and scaled rate stationarity are checked. Local precision and model-mismatch flags must be reported. Scores with estimated rates and selected masks are heuristic; no familywise chemical-detection guarantee is claimed.

## Method-specific formulations in manuscript-v4

The [editable method section](../paper/sections/methods.tex), [mathematical appendix](../paper/sections/method_details.tex), and [27-file equation/source map](../paper/evidence/method_formulations.json) specify operators, unknowns, constraints, objectives, updates, regularization, selection and numerical limits. Algorithm 1 is an actual numbered `algorithm`/`algpseudocode` float with inputs, outputs, loops, eligibility checks, selection and final refitting. Its equations are linked in the pseudocode.

Here n denotes acquisitions, p retained frequencies, q diffusion cells, and r shared factors. A basis operator integrates exp(-b D) against unit-integral positive functions. Its coefficients are masses. Atomic rates instead remain continuous; binning them for display is not a grid inversion.

| Method | Specific inversion and selector |
|---|---|
| TRAIn supplied file | h=eta²; minimize ||K h-y||². Gauss-Newton B=8 diag(eta) KᵀK diag(eta), truncated-CG trust step, actual/predicted reduction ratio. Stop relative to an NNLS residual floor. The core has no explicit Tikhonov penalty. |
| Native atomic RAI | Profile A≥0 out of ||(K(d)A-Y)/sigma||²/2. Bounded L-BFGS-B in log d with ten starts and residual births. Native score RSS/sigma²+r(p+1)log(np). |
| Density RAI | Positive bin masses C; data loss/(2n) + lambda_s||R C||²/2 + eta tr(CᵀH C)/2 + gamma tr(H C L_graph Cᵀ)/2. R acts on density differences; H=diag(1/bin width). Positive ridge gives a unique fixed-grid minimizer. Validation decides grid splits. |
| Adaptive Haar RAI | Positive normalized leaf masses U, physical masses P_tree U. Data loss/(2n)+eta||U||²/2+lambda sqrt(p) sum of Haar-detail row norms. ADMM uses positivity projection and group shrinkage; feasible primal/dual gap checks each fixed tree. Completed penalty paths compete on validation. |
| RAI-Net original / revised | Two-layer tanh network ranks tree actions using 14 specified features. Smooth-L1 training on signed-log gains; original target is training-objective gain, revised target is internal-validation prediction gain. It retains the Haar objective and numerical acceptance rule; it is not an r classifier or direct density regressor. |
| Native DOME | Same atomic residual objective, active-subset NNLS and full residual-Jacobian trust-region refinement. Curvature, spectral contrasts and moment-preserving splits/exchanges propose rates. Native corrected parameter-count score selects order. |
| RAI-S / DOME-S | Retain their respective proposal paths; screen candidate frequency supports, refine each, compare held-out predictions and prefer fewer rates then fewer allowed amplitudes within the paired allowance. Equation and Algorithm 1 specify the exact order of operations. |
| TRAIn-MF | Shared unit-mass density profiles S, spectra A≥0; objective and true MATLAB NNLS above. Joint simplex QP for S, then profiled SQP as needed. Rank evidence limits candidate search; empirical paired row-loss standard errors select predictive factors. |
| DOME averaging | Fixed native rates; NNLS on every spectral subset, entropy-regularized soft weights with scores RSS/sigma²+6 subset size and temperature 0.5. Different from DOME-S. |
| RAI-FLEX | Y≈[K(d),G C]A, unit-mass positive spline profiles C, second-derivative penalty lambda||L₂C||²/2. Profiled NNLS, log-rate/softmax optimization then simplex refinement. Heuristic local-effective-dimension score selects atom/profile allocation. |
| Partial-C | C=s a+V≥0 with one unit-mass shared spline profile; total-density first-derivative and private-density L² penalties. Joint conditional NNLS for (a,V), simplex outer SLSQP. Shared rank is fixed at one in the comparator. |
| CIRCE adapters v1 / v2 | Positive convex quadratics with density roughness and spectral graph. V2 constrains C=T Z, Z≥0, using a fixed positive mass-preserving decoder and column-scaled roughness. FISTA or strict cyclic NNLS; residual-cone/mass-cap feasibility is checked afterwards. No CIRCE-Net inference or alternative-measure bounds are attributed to these adapters. |

Run `python paper/scripts/check_formulations.py` for 20 deterministic finite-dimensional algebra checks (gradients, QR/Kronecker identity, feasible Haar gap, DOME moments and decoder mass conservation). This does not rerun fits or establish statistical recovery. Results are recorded in `verification/formulation_checks.json`.

### Distinctions that affect interpretation

- TRAIn-MFV31's `lambdaSparse` augments the design by sqrt(lambda) diag(sqrt(w)); the resulting term is weighted quadratic, not an explicit L1 penalty. Clipped residual pseudo-curves and adaptive smoothing/weights do not implement the current fixed TRAIn-MF objective.
- TRAIn-MF's empirical paired standard error across validation acquisitions differs from RAI-S/DOME-S's conditional Gaussian prediction-distance allowance. Neither is a calibrated chemical order test after adaptive masking and selection.
- A fixed convex subproblem can have a unique minimizer while its selected tree, grid, factors or underlying species remain nonidentifiable. Squared positivity can have a zero gradient at a point failing NNLS KKT.
- The public API scope remains the packaged methods listed above in the reproduction section. Formulating historical variants does not add runnable implementations or retraining to this curated release. Numerical software, saved estimates, metrics and figures remain unchanged.

## Identifiability
Finite Laplace observations do not uniquely identify arbitrary positive measures. Narrow close components, weak species, proportional spectra, baseline/phase errors, noise misspecification and overly narrow bounds can defeat recovery. A fine grid cannot manufacture information. Recovering the fitted rank is weaker than recovering all diffusion rates and weaker still than chemical identification. The manuscript reports these separately.

## Reproduction scopes
- examples: three fixed development examples; regenerate with scripts/make_examples.py.
- benchmarks/run_atomic.py: exact historical 48-case test generator and four atomic methods; 36 signal-present discrete problems, six nulls, six broad misspecification controls. Its historical mask uses all outer training rows, including internal validation rows.
- benchmarks/frozen/metrics.csv: full existing per-case metrics including archived and exploratory neural results. No frozen numerical results were altered for the public release.
- paper/scripts/prepare_evidence.py: regenerate manuscript tables and figures from a minimal saved synthetic fixture set, with no solver calls.
- MATLAB example and optional API backend: actual native TRAIn-MF fit using licensed MATLAB. The manuscript's original TRAIn-MF figure remains the hashed saved output.
- C#: typed SDK/CLI for the same API. It is not an independent numerical reimplementation and provides no third independent scientific validation.
Neural training and the full historical multi-algorithm survey are outside this curated release. Their frozen reported metrics are retained; the release does not claim end-to-end retraining of those exploratory models.

