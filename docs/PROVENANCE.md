# Provenance / Procedencia
The original TRAIn algorithm is due to **K. Xu and S. Zhang**, *Analytical Chemistry* 2014, 86, 592–599, [doi:10.1021/ac402698h](https://doi.org/10.1021/ac402698h).

**Francisco M. Arrabal-Campos** developed **TRAIn-MF**, its multifrequency extension, through successive revisions of the same method. V2 refers to an earlier V1; V3 consolidates the preceding implementation work; V3.1 refines V3. V3 is a consolidated pre-publication implementation, not a separate method against which TRAIn-MF is proposed. The corresponding author describes approximately one year of research and development. The [development history](TRAIN_MF_HISTORY.md) and [source inventory](../provenance/train_mf_development_history.json) distinguish this author-reported duration from what the surviving source headers and file timestamps establish.

The public source `matlab/legacy/TRAIn_DOSY_MFV31.m` is preserved byte-for-byte and matches the author's V3.1 file. Its header identifies Universidad de Almería, NMRMBC Research Group, version 3.1 (January 2026). The header date is source metadata, not an independently verified public release date. This revision alternates nonnegative spectral amplitudes and diffusion profiles. `train_component_traincore` calls `local_train_core_kernel`, which uses the squared variable h=eta², a Gauss–Newton model, truncated conjugate gradients, and actual/predicted reduction to adjust a trust radius. These operations document the derivation from TRAIn.

The implementation in `matlab/reference` continues the development of **the same TRAIn-MF method**, with unit-mass profile columns, a stated common penalized objective, MATLAB `lsqnonneg` amplitude steps, simplex-constrained profile steps, profile refinement and numerical convergence gates. These are substantive numerical refinements: the constrained updates replace the embedded TRAIn iteration of the preserved V3.1. Common method identity does not imply identical optimization steps or results across versions. The archive directory name `legacy` denotes a preserved source revision, not a different algorithm or separate authorship.

RAI-S and DOME-S have different proposal paths followed by the same support/prediction estimator. They are not descendants of TRAIn merely because scipy.optimize.least_squares uses a trust-region method.

## Source preservation
provenance/source_hashes.json records SHA-256 of each copied numerical source, the preserved TRAIn-MF V3.1 revision, fixed metrics and generator. src/train_dosy/_frozen/support.py and the native MATLAB reference are unchanged. Only the RAI package initializer is reduced to avoid importing unused neural code; model.py, solver.py and components.py remain exact copies. No proprietary MATLAB NNLS source, raw NMR experiments, account credentials or third-party algorithm archives are included.

The initial public release is 0.1.0 (2026-09-25). Source snapshots come from the author's NEW_ITERATIVE_ILT and DOSY development projects. This release adds packaging, a validated data contract, clients, examples and publication materials; it does not rerun or silently replace the frozen manuscript benchmark.

## Español
TRAIn original corresponde a Xu y Zhang. **TRAIn-MF es la extensión multifrecuencia desarrollada por Francisco M. Arrabal-Campos**. V2, la V3 depurada previa al artículo y su mejora V3.1 son revisiones sucesivas del mismo método. V2 menciona una V1, pero no se ha localizado una copia identificable de esa primera versión en el archivo revisado. El autor sitúa el trabajo en aproximadamente un año de investigación. Las cabeceras, fechas y hashes conservados se detallan en el [historial](TRAIN_MF_HISTORY.md); no prueban por sí solos la duración completa.

La implementación actual de TRAIn-MF continúa ese desarrollo y revisa la optimización mediante NNLS y restricciones simplex. Se especifican los cambios matemáticos para reproducir cada implementación. El archivo V3.1 conserva sus opciones originales y su núcleo derivado de TRAIn; los resultados de la implementación actual no se atribuyen automáticamente a todas las versiones previas. Esta aclaración editorial no modifica código ni resultados.

No se ha fijado el rango verdadero de los ejemplos en los ajustes. La verdad se conserva aparte y se utiliza únicamente para evaluar y dibujar. Las limitaciones de identificación se mantienen explícitas.

## Dated validation supplement

The 17 October 2026 validation sources are separate from the API and archived primary benchmark. They preserve their numerical parameters and use explicit source snapshots. The runtime-monitored regularized TRAIn-MF experiment is not the signed-v2.2 backend. The public supplement excludes industrial spectra, client reports, editorial correspondence and response letters. See VALIDATION_20261017_EN.md for dependencies and measured limitations.
