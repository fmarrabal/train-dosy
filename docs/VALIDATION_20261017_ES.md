# Suplemento de validación: 17 de octubre de 2026

Este suplemento recoge lo que mostraron los nuevos experimentos, incluidos los fallos y los casos en los que un método de referencia obtuvo mejores resultados. Conserva la API 0.2.1, su backend TRAIn-MF signed-v2.2 y los resultados originales de MF-AUTO. La fecha de las carpetas identifica el estudio; no acredita una publicación editorial ni una nueva versión del software.

## Resultados y límites

| Estudio | Diseño e independencia | Conclusión que permiten los datos |
|---|---|---|
| Selección con ruido puro | 300 matrices; 100 por cada uno de tres niveles de ruido, compartidas por ambos métodos | Amplía los seis controles originales. Hay que informar los estratos de ruido por separado; no demuestra un control general de falsos descubrimientos. |
| Componentes débiles | 96 matrices: cuatro amplitudes relativas, tres SNR y ocho repeticiones | La tasa débil se recuperó en 45/96 ajustes RAI-S y 44/96 DOME-S. La coherencia espectral no hace detectables todos los componentes débiles. |
| Ablación del algoritmo | 36 matrices con señal y combinaciones pareadas de cribado/selección de orden | Los pasos compartidos importan. La coincidencia entre RAI-S y DOME-S no demuestra dos mejoras independientes. |
| Convergencia de TRAIn-MF regularizado | Nueve matrices rango/SNR; cuatro límites de barridos previos y dos inicios por ajuste | 14/36 ajustes seleccionados alcanzaron KKT relativo <= 1e-6; 22 quedaron sin resolver dentro del presupuesto común. El rango se proporciona para aislar la optimización. No valida el selector automático ni prueba que ocho barridos basten. |
| Política neuronal | 120 problemas nuevos, cinco semillas de entrenamiento y una política clásica: 720 ajustes | En 105 problemas con señal, el cambio medio de error de transporte fue -0,00054 (intervalo pareado [-0,00174; 0,00056]); el de predicción limpia en los 120 fue -0,00012 ([-0,00044; 0,00010]). No se demuestra ventaja predictiva. Los ajustes finales cumplieron las comprobaciones numéricas declaradas. |
| Referencias clásicas | 47 matrices con ruido independiente y 24 casos de distorsión (12 verdades bajo dos distorsiones), siete métodos | Hay 504 ajustes guardados y 497 analizados: el mismo piloto se excluye en los siete métodos. Se comparan SCORE-CV, SCORE-NNLS-CV, LOCODOSY, Tikhonov positivo y el objetivo PALMA con sus envoltorios declarados. PALMA no cumplió el certificado de convergencia de todas las columnas dentro del presupuesto; se informa como resultado con presupuesto finito. |
| Resinas industriales | Cuatro series, tres tiempos de difusión por serie, cinco métodos: 60 ajustes | Son 12 bloques agrupados en cuatro series, no 12 mezclas independientes. RAI-S y DOME-S alcanzaron el límite de cuatro componentes y conservaron desajuste del modelo. Solo se publican diagnósticos escalares; no hay verdades independientes sobre el número de sustancias ni sus D. |
| Ejemplo público de DOSY Toolbox | 30 adquisiciones, 4074 canales seleccionados y siete adquisiciones externas | RMSE predictivo relativo: 0,582% RAI-S/DOME-S; 0,232% SCORE-CV con signo libre; 0,582% SCORE-NNLS-CV; 0,270% LOCODOSY local con signo libre. SCORE convergió y predijo mejor este ejemplo. Sus amplitudes con signo no representan una medida positiva; no se dispone de D independientes. |

Los intervalos neuronales son bootstrap descriptivos y pareados. Las semillas y los ajustes repetidos no aumentan el número de problemas de prueba independientes. Tampoco lo hacen los distintos tiempos de difusión de una misma resina. Se conservan los indicadores de fallo.

La entrada monitorizada de MF es `validation/mf_convergence/instrumented/mf_robust/DOSY_MF_Runtime.m`, relativa a la carpeta del estudio. Mantiene el objetivo de la referencia regularizada 5.1 y comprueba durante la ejecución la condición de curvatura especificada. La condición se refiere a un bloque de optimización y no demuestra identificación química. La ilustración histórica usó la referencia 5.0. Ninguna se presenta como si fuera el backend signed-v2.2 de la API.

## Archivos y reproducción

Los comandos de la API y del benchmark original siguen en el README principal. Los nuevos materiales están en `paper/revision_complete_20261017/validation`; el primer estudio de nulos, ablación y distorsiones conserva `paper/revision_20261017/validation` porque sus importaciones dependen de esa ruta. Cada estudio tiene protocolo y resumen. `PUBLIC_VALIDATION_MANIFEST.json` recoge los archivos públicos y sus hashes.

Se publican código, protocolos, resultados compactos JSON/CSV, cinco redes entrenadas con sus historiales, trazas sintéticas de entrenamiento y las nueve matrices pequeñas de entrada del estudio MF. Se excluyen las grandes matrices ajustadas, los datos industriales y los documentos privados de laboratorio o editoriales. Los resúmenes se pueden inspeccionar directamente. Para volver a comprobar cada predicción hay que repetir los cálculos y generar las matrices omitidas; un JSON compacto no las sustituye.

Use entornos separados: el estudio atómico/experimental registra Python 3.13.13, NumPy 2.5.1 y SciPy 1.18.0; el neuronal registra Python 3.13.13, NumPy 2.5.3, SciPy 1.18.1 y PyTorch 2.14.0 para CPU. Los archivos `requirements_revision_atomic.txt` y `requirements_revision_neural.txt` fijan las dependencias completas. MATLAB se ejecutó con R2026a y Optimization Toolbox. No hace falta GPU. El sistema operativo y las bibliotecas pueden afectar al redondeo y a los tiempos.

El benchmark externo necesita `diffusion-inversion` junto a la carpeta `train-dosy`. Desde el directorio que contiene ambas:

```sh
git clone https://github.com/fmarrabal/diffusion-inversion.git
git -C diffusion-inversion checkout 2aecc92fd74f809f96f05d36f9da0f3a5ca512bd
```

Esta revisión conserva la traducción atribuida de SCORE/LOCODOSY. La variante SCORE con NNLS se identifica por separado de la referencia con amplitudes de signo libre. PALMA designa aquí el objetivo, implementación y selección declarados; no implica identidad de código con el programa original.

Para recalcular solo agregados, trabaje en una copia desechable del árbol publicado, pues los agregadores escriben sus salidas. Estos comandos no ajustan modelos:

```sh
python paper/revision_20261017/validation/summarize_revision_validation.py
python paper/revision_complete_20261017/validation/check_compact_summaries.py
python paper/revision_complete_20261017/validation/external/analyse_external.py
```

Para ajustes atómicos nuevos, cree otra copia con las mismas rutas relativas, los scripts y las dependencias, dejando ausentes las carpetas generadas `results`/`inputs` del estudio. Ejecute el script correspondiente: `run_frozen_additional_validation.py`, `run_frozen_distortion_stress.py`, `run_weak_power.py` o `run_external_benchmark.py`. `run_score_nnls.py` añade la comparación SCORE de amplitudes positivas. No reutilice resultados antiguos: varios programas reanudan los casos por su nombre de archivo. Son cálculos locales de investigación, no trabajos de CI.

Para una repetición neuronal independiente:

```sh
python paper/revision_complete_20261017/validation/neural_ablation/prepare_portable_replay.py --destination neural-repeat
```

El destino debe estar vacío. El adaptador solo cambia rutas; mantiene semillas, parámetros y reglas de inferencia. `neural-repeat/replay_adaptation.json` indica los comandos posteriores de entrenamiento, comprobación, inferencia y análisis. Instale allí el entorno neuronal separado. Las cinco redes guardadas se pueden inspeccionar sin entrenar. El factorial se ejecuta con `factorial_study.py` en su ruta original dentro del repositorio.

Para MF, prepare un directorio nuevo con los scripts `.m`/`.py` y las carpetas `source_frozen` e `instrumented`, sin configuración ni resultados generados. En MATLAB ejecute `prepare_mf_experiment`, `test_runtime_curvature`, `run_mf_batch([1 2 3])` y `audit_completed_outputs`; después, `aggregate_mf.py` en Python. `EVIDENCE_EN.txt` detalla objetivo, tolerancias y comprobaciones. La publicación compacta conserva los JSON; las matrices MAT que exige la auditoría de predicciones se regeneran en la repetición. Los hashes originales identifican la ejecución original; el nuevo directorio tiene su propio registro de procedencia.

## Datos experimentales y licencias

El programa de resinas se publica para inspección, pero sus observaciones no se distribuyen. Este subconjunto público no basta para repetir aquellos ajustes. `summary_public.json` y `metrics.csv` mantienen los números y diagnósticos, sin los cruces privados de identidad de los materiales.

El ejemplo Varian procede del archivo oficial DOSY Toolbox del Manchester NMR Methodology Group. La URL exacta y el SHA-256 están en `experimental/public_mixture/toolbox_legacy_download.json`. Desde esa carpeta, con el entorno atómico:

```sh
python download_varian.py
python prepare_varian.py
python run_varian.py
python summarize_varian.py
```

La descarga debe preceder a la auditoría de hashes de los datos crudos. El procesado conserva la partición fijada y estima una fase común solo con las adquisiciones de descubrimiento. Ajusta todos los canales del soporte de absorción positiva detectado en entrenamiento; fuera del soporte no se estima señal y no se demuestra ausencia. Ni la fase ni el escalado del kernel usan respuestas externas o identidad química ajustada. El ejemplo aporta evidencia predictiva, sin coeficientes de difusión medidos independientemente. El archivo histórico no incluye una licencia de datos explícita; por eso se publica el descargador con la atribución, sin redistribuir los datos medidos.

El código es GPL-3.0-or-later salvo los avisos de terceros conservados. Deben acompañarlo `NOTICE`, el aviso GPL de GNAT, las atribuciones SCORE/LOCODOSY y la licencia BSD de nmrglue. La dependencia fijada `diffusion-inversion` tiene `THIRD_PARTY_NOTICES.md`, `licenses/` y `docs/LOCODOSY_SOURCE_NOTICE.md`. MATLAB y `lsqnonneg` se licencian aparte y no se redistribuyen. No se añade ni se activa ningún workflow de GitHub Actions.

`EXCLUDED_SCIENTIFIC_ARTIFACTS.json` identifica los artefactos sintéticos y del ejemplo público omitidos mediante ruta, tamaño y SHA-256. Registra evidencia local, no enlaces de descarga de una release. Las observaciones industriales privadas quedan fuera tanto del inventario como del paquete público. Se incluyen las 80 trazas sintéticas de entrenamiento, las nueve entradas MF y las cinco redes entrenadas.
