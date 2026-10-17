# Third-party notices for the dated validation supplement

The repository's GPL-3.0-or-later licence does not replace the licences below.
Keep the original notices with redistributed source. No proprietary MATLAB
NNLS source, industrial observations or private documents are redistributed.

* SCORE/LOCODOSY: `paper/revision_complete_20261017/validation/experimental/vendor/locodosy_native.py`
  is the attributed Python translation of the GNAT implementation, with original
  MATLAB attribution to Mathias Nilsson (2007, 2010, 2017). The module retains
  its GPL-3.0-or-later translation notice. The full GNAT GPL is adjacent as
  `COPYING-GNAT`; `ATTRIBUTION.txt` explains the translation and CV wrappers.
* nmrglue 0.11: source is under `experimental/public_mixture/vendor_runtime/`,
  relative to the dated validation directory. Copyright (c) 2010-2015 Jonathan
  J. Helmus. The complete BSD licence and distribution metadata are retained
  in `nmrglue-0.11.dist-info/`. The numerical source is unchanged; the reader
  loader avoids importing an unrelated NumPy-incompatible module.
* diffusion-inversion: external studies use the separately public commit
  `2aecc92fd74f809f96f05d36f9da0f3a5ca512bd`. Its `THIRD_PARTY_NOTICES.md`,
  `LICENSE`, `licenses/`, `docs/LOCODOSY_SOURCE_NOTICE.md` and GNAT `COPYING`
  govern its source. The main train-dosy repository does not silently absorb
  that dependency. Follow the sibling-checkout instructions in the validation
  guide. The staging dependency snapshot is for an explicitly labelled source
  bundle if one is made; it is not another train-dosy solver implementation.
* NumPy, SciPy, Matplotlib, PyTorch and the API libraries are separately installed
  dependencies; their pinned versions and their own licences apply.
* Original TRAIn (Xu and Zhang, 2014, DOI 10.1021/ac402698h), NNLS (Lawson and
  Hanson), variable projection (Golub and Pereyra) and the named classical
  comparators retain algorithmic credit. A mathematical citation does not
  substitute for a software licence; no restricted upstream PALMA distribution
  is copied in this supplement.
* The Manchester DOSY Toolbox legacy data are retrieved from the official URL
  and checked against the recorded hash. No explicit data licence was found
  in that archive, so measured arrays are not included. See its ATTRIBUTION.txt.

The historical RAI/DOME sources and synthetic training traces were supplied
from the project owner's research projects. Their source snapshots and hashes
are retained. Existing MDPI LaTeX template notices remain applicable to the
manuscript source when that source is distributed.
