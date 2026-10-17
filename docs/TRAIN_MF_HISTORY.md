# Development history of TRAIn-MF

## Español

**TRAIn-MF y TRAIn-DOSY-MF designan el mismo método multifrecuencia.** Los sufijos V2, V3 y V3.1 identifican etapas de su desarrollo. V3 es la versión depurada previa al artículo y V3.1 una mejora de esa versión. La implementación actual con restricciones continúa este trabajo. El TRAIn original de Xu y Zhang es el antecedente del que parte la extensión multifrecuencia de Francisco M. Arrabal-Campos.

El autor indica aproximadamente **un año de investigación y desarrollo**. Esta duración se recoge como información aportada por el autor el 26 de septiembre de 2026. El archivo permite reconstruir revisiones conservadas, pero sus fechas no constituyen un registro completo del inicio y fin del proyecto.

| Etapa | Evidencia conservada | Última modificación de la copia, hora de Madrid |
|---|---|---|
| V1 | La cabecera de V2 menciona expresamente problemas corregidos respecto a V1. No se ha localizado un archivo MF V1 identificable. | No establecida |
| V2 | `TRAIn_DOSY_MFV2.m`, cabecera e `info.version` 2.0. Declara ajustes de inicialización, pesos, componentes degenerados y selección opcional de rango. | 18 enero 2026, 00:46:55 CET |
| V3 | `TRAIn_DOSY_MF.m` y `TRAIn_DOSY_MF_V3.m`, cabecera V3 e `info.version` 3.0. Declara la consolidación de la estabilidad de «MF_fixed» y los diagnósticos de MFV2. | 18 enero 2026, 00:56:18 y 00:56:27 CET |
| V3.1 | `TRAIn_DOSY_MFV31.m`, cabecera e `info.version` 3.1. Declara refinamientos de validación, reinicialización, suavizado y selección, incluida una corrección en `init_nmf_style`. | 18 enero 2026, 19:35:54 CET |
| Implementación empleada en el artículo | `matlab/reference` conserva la formulación con objetivo común, NNLS de MATLAB, restricciones simplex y comprobaciones de convergencia. Sigue siendo TRAIn-MF. | Software publicado como v0.1.0 el 25 septiembre 2026 |

Los dos archivos V3 son **idénticos byte a byte**: SHA-256 `9fa86991541ac8f57cddcaac9ca1ef1a56e3065dddca49074a4edefbdfb9fdf1`. Por tanto, el nombre sin sufijo `TRAIn_DOSY_MF.m` no permite identificar una V1 adicional. V3.1 tiene SHA-256 `d7d2ec44b70f3f68baeb4f7c57592c3e14067cb1a5ed18244d5648724ef4a82f` y coincide con el archivo V3.1 preservado en el repositorio público.

Las fechas de creación del sistema de archivos son de mayo de 2026 y posteriores a las de modificación de enero, lo que es compatible con una copia del directorio. No se utilizan como fechas de nacimiento del algoritmo. La cabecera de `TRAIn_Improved.m` contiene «2025», pero ese archivo pertenece al contexto de desarrollo de TRAIn y no identifica por sí solo una MF V1 fechada. Algunos documentos Word tienen metadatos internos de 2013 que parecen heredados del documento base; tampoco se utilizan para fechar el desarrollo.

La descripción de las mejoras de cada cabecera documenta la intención de esa revisión; no sustituye una validación numérica. La revisión editorial no ha cambiado códigos, datos, figuras ni resultados. El artículo mantiene la explicación de las diferencias entre las actualizaciones V3.1 y la optimización actual dentro del mismo método.

El [inventario JSON](../provenance/train_mf_development_history.json) incluye hashes, tamaños, fechas UTC y líneas de cabecera. V2 y V3 se inspeccionaron en el archivo local indicado por el autor; la fuente V3.1 sí está incluida en la distribución pública. La [procedencia general](PROVENANCE.md) recoge las atribuciones y el alcance de la distribución.

## English

TRAIn-MF and TRAIn-DOSY-MF refer to the same multifrequency method. V2, the consolidated pre-publication V3, and its V3.1 refinement are successive development versions, not separate inversion methods. The current constrained implementation continues this development while making its objective, conditional updates and convergence checks explicit. Original TRAIn authorship remains attributed to Xu and Zhang; the multifrequency extension is due to Francisco M. Arrabal-Campos.

The corresponding author reports approximately one year of research and development. The surviving files document January 2026 source revisions and later implementation work; filesystem timestamps alone do not establish the full duration. V1 is referenced in the V2 header, but a separately identifiable V1 file was not located. The two V3 filenames contain identical bytes, and the public V3.1 source matches the author's archive exactly.

The table above gives the local modification times of the surviving copies, not first implementation or public release dates. The machine-readable inventory preserves UTC timestamps, source-header evidence and SHA-256 hashes. Creation dates after modification dates and inherited Word template metadata are not treated as development milestones. Stated header improvements are historical descriptions, not newly validated performance claims. This editorial clarification leaves every numerical source and frozen result unchanged.
