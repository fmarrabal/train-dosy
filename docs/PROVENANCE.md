# Method and software attribution / Atribución

Original TRAIn: K. Xu and S. Zhang, *Analytical Chemistry* 2014, 86, 592–599, [DOI](https://doi.org/10.1021/ac402698h).

TRAIn-MF is the multifrequency extension developed by Francisco M. Arrabal-Campos. Its shared-profile factorization is described in the manuscript. The signed-data trust-region implementation and the regularized constrained reference have explicitly different optimization and selection protocols. Both are named in the documentation by the operation they execute.

RAI-S and DOME-S use different proposal paths with the same final spectral-support and predictive-order estimator. Their Python numerical sources and the reference MATLAB implementation are supplied with the study protocols. The C# library is a typed API client.

NNLS follows Lawson and Hanson; variable projection follows Golub and Pereyra. SCORE/LOCODOSY translations retain GNAT attribution and GPL notices. Third-party conditions are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The MDPI class retains its own notices. Proprietary MATLAB implementation sources are not redistributed.

Earlier software and manuscript revisions remain in Git history and the existing tagged releases. The current tree contains the canonical final files and the evidence needed to interpret the reported studies. Reorganized paths do not represent newly estimated results.

## Español

TRAIn corresponde a Xu y Zhang. TRAIn-MF es la extensión multifrecuencia de Francisco M. Arrabal-Campos. Se distinguen el protocolo con observaciones de signo y la referencia regularizada del artículo. RAI-S y DOME-S comparten el estimador final de soporte/orden, con mecanismos de propuesta diferentes. C# actúa como cliente de la API.

Las licencias y atribuciones se conservan con cada implementación. Los resultados guardados y las entradas sintéticas no cambian al organizar el repositorio; los antecedentes permanecen recuperables en el historial de Git.
