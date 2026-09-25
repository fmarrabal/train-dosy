# TRAIn-DOSY 0.1.0 / Public research release

## English
- Preserved TRAIn_DOSY_MFV31 by Francisco M. Arrabal-Campos, with original TRAIn attribution to Xu and Zhang.
- Unchanged native MATLAB MF reference, 256+ bins, installed MATLAB NNLS and automatic predictive factor selection.
- Python RAI-S/DOME-S, CLI and documented REST API; optional native MATLAB MF backend.
- C#/.NET 10 typed SDK and CLI, invoking the reference API.
- English/Spanish READMEs and API/mathematical documentation.
- One-, two- and three-component synthetic examples with separate ground truth, results and DOSY plots.
- Revised 22-page Mathematics draft, 23 bibliography entries and public repository citation.
- Local verification: 14 Python tests; six demonstration fits; 12 atomic benchmark smoke fits reproducing the archived clean-prediction metric exactly; MATLAB direct/API agreement; C# and MATLAB HTTP client prediction agreement; .NET build; installable Python wheel.

This is research software. C# is a client, not an independent numerical estimator. The full historical 48-case benchmark was not rerun for this packaging release; its frozen results and runnable atomic protocol are supplied. Neural retraining and real experimental validation are not claimed. Author metadata and declarations in the paper remain pending. MATLAB is separately licensed. GitHub Actions remain disabled.

## Español
Se conserva tu MFV31 y se atribuye su procedencia al TRAIn original. Se publica el MF nativo de MATLAB sin cambiar su núcleo, RAI-S/DOME-S en Python, API, SDK/CLI C#, ejemplos con verdad de referencia y documentación completa en ambos idiomas. El artículo incorpora seis referencias tuyas adicionales, MFV31 y la dirección pública.

Las comprobaciones son locales: 14 pruebas Python, seis ejemplos, 12 ajustes de comprobación del benchmark, MF nativo/API y clientes MATLAB/C#. No se presenta C# como otro solver independiente ni se afirma haber repetido el benchmark completo, reentrenado las redes o validado experimentos reales en esta publicación. La portada y declaraciones siguen pendientes de los autores.
