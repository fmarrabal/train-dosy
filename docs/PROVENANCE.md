# Provenance / Procedencia
The original TRAIn algorithm is due to **K. Xu and S. Zhang**, *Analytical Chemistry* 2014, 86, 592–599, [doi:10.1021/ac402698h](https://doi.org/10.1021/ac402698h).

**Francisco M. Arrabal-Campos** authored the multifrequency extension archived as matlab/legacy/TRAIn_DOSY_MFV31.m. Its header identifies Universidad de Almería, NMRMBC Research Group, version 3.1 (January 2026). The file is preserved byte-for-byte. The header date is source metadata, not independently verified release history. The extension alternates nonnegative spectral amplitudes and diffusion profiles; train_component_traincore calls local_train_core_kernel. That kernel uses the squared variable h=eta², a Gauss–Newton model, truncated conjugate gradients, and actual/predicted reduction to adjust a trust radius. This supports a direct TRAIn algorithmic lineage, rather than a resemblance based only on a generic trust-region solver.

The later reference in matlab/reference is a **numerical revision of that TRAIn-MF research line**: unit-mass profile columns, a stated common penalized objective, MATLAB lsqnonneg amplitude steps, simplex-constrained profile steps, profile refinement and numerical convergence gates. Its optimization is not the original embedded TRAIn iteration. This difference does not erase authorship or lineage.

RAI-S and DOME-S have different proposal paths followed by the same support/prediction estimator. They are not descendants of TRAIn merely because scipy.optimize.least_squares uses a trust-region method.

## Source preservation
provenance/source_hashes.json records SHA-256 of each copied numerical source, the TRAIn-MF ancestor, fixed metrics and generator. src/train_dosy/_frozen/support.py and the native MATLAB reference are unchanged. Only the RAI package initializer is reduced to avoid importing unused neural code; model.py, solver.py and components.py remain exact copies. No proprietary MATLAB NNLS source, raw NMR experiments, account credentials or third-party algorithm archives are included.

The initial public release is 0.1.0 (2026-09-25). Source snapshots come from the author's NEW_ITERATIVE_ILT and DOSY development projects. This release adds packaging, a validated data contract, clients, examples and publication materials; it does not rerun or silently replace the frozen manuscript benchmark.

## Español
TRAIn original corresponde a Xu y Zhang. La extensión multifrecuencia TRAIn_DOSY_MFV31 corresponde a Francisco M. Arrabal-Campos y conserva dentro el núcleo de trust-region derivado de TRAIn. El TRAIn-MF posterior revisa la optimización mediante NNLS y restricciones simplex; no debe confundirse con una implementación idéntica del núcleo histórico. El código histórico conserva sus opciones antiguas y no es el estimador validado por las pruebas de esta versión.

No se ha fijado el rango verdadero de los ejemplos en los ajustes. La verdad se conserva aparte y se utiliza únicamente para evaluar y dibujar. Las limitaciones de identificación se mantienen explícitas.

