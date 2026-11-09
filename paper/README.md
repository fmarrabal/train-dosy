# Final manuscript

- [Manuscript.pdf](Manuscript.pdf): the final scientific text.
- [manuscript/Manuscript.tex](manuscript/Manuscript.tex): editable source.
- `manuscript/Definitions/` and `manuscript/figures/`: the required template and figures.

Compile locally with `./paper/build.ps1`, or run `pdflatex` three times on `Manuscript.tex` from `paper/manuscript/`, using a separate output directory. No numerical inverse fitting or training is needed to compile the paper.

The synthetic inputs and figure-replay script are under `validation/primary/`. The other validation folders hold the reported protocols, compact outcomes and checkpoints. Industrial observations and private editorial documents are supplied through the journal supplement rather than this repository.

Publisher-controlled article identifiers and dates remain as supplied in the proof. The journal assigns the final publication identifiers.

## Español

El PDF y su fuente editable son los archivos vigentes del manuscrito. El código de validación, los datos sintéticos y los resultados que sustentan el artículo se organizan en `validation/`. Las versiones anteriores quedan en el historial de Git.
