# RAI-S y DOME-S: selección de soporte espectral

Versión de investigación para DOSY discreto. Conserva los originales y MF automático. El sufijo **S** identifica una nueva selección del soporte espectral, compartida por RAI y DOME; no es una nueva ley física ni una afirmación de superioridad universal.

## Corrección de la figura MF y selección automática explícita

El ajuste MF original no cambió: su SHA-256 coincide con el de la galería anterior. Sin embargo, la primera figura de esta entrega remuestreaba sus masas a otra rejilla (cambio relativo de la matriz representada: 1.89%) y usaba un umbral absoluto unas 12.4 veces menor. Eso alteraba su aspecto. La figura corregida conserva cada masa MF en su celda original y restaura la escala común y el umbral del 0.1% de la galería anterior. No se ha recalculado MF ni eliminado ninguna de sus amplitudes. Los archivos previos se conservan en `correction_20260925/before/` y en el ZIP anterior.

**El número de componentes es automático.** No se introduce r ni r verdadero. La salida MATLAB `r_auto` indica el valor elegido; `candidate_ranks` contiene los órdenes comparados y `selection_mode` vale `automatic`. La búsqueda actual compara 0–4 componentes: cuatro es un límite computacional, no un número impuesto. `component_search_limit_reached` avisa si el resultado llega al límite. El criterio usa predicción sobre gradientes reservados y prefiere menos tasas cuando la mejora no justifica añadir otra. Se ha comprobado de nuevo con entradas que contienen solo Y, b, ppm, sigma y máscara: ambos métodos eligen 1, 2 y 3 en los tres casos de control, sin modificar los ajustes validados.

## Qué cambia

RAI y DOME ajustaban tasas comunes y una amplitud no negativa por frecuencia. La restricción de no negatividad deja pequeñas amplitudes positivas al ajustar ruido: una tasa física real aparece entonces en regiones donde esa especie no tiene señal. Recortar la altura relativa al pico mayor también borraría especies minoritarias reales.

1. Se reservan gradientes internos, distribuidos por el rango de b.
2. RAI o DOME generan candidatos con 0–4 tasas sobre los demás gradientes.
3. Para cada tasa y ventana espectral contigua, se agrega evidencia respecto al ruido después de proyectar las otras tasas. Se generan candidatos con soporte restringido, además de conservar los originales.
4. Se evalúan sus predicciones en los gradientes reservados. Entre candidatos compatibles con la regla predictiva de una desviación estándar, se prefieren menos tasas y después menor soporte espectral.
5. Se reajustan tasas y amplitudes con NNLS, usando todos los gradientes de entrenamiento y el soporte elegido. Las amplitudes excluidas son exactamente cero en el resultado, no solo invisibles en el gráfico.

Las tasas son continuas, fuera de rejilla. La exportación DOSY conserva masa en **256 bins logarítmicos**. El ajuste utiliza todas las frecuencias seleccionadas conjuntamente; no una frecuencia aislada.

## Modelo y límites matemáticos

Se minimiza `0.5 * ||(exp(-b D) A - Y)/sigma||_F^2`, con `A >= 0`, límites positivos de D y ceros estructurales en el soporte elegido. Se elimina A por NNLS y se optimiza log D mediante proyección variable y trust region. Los jacobianos incluyen el término de variación del ajuste lineal. Se verifican KKT para A y estacionariedad local para D; **no se demuestra optimalidad global**.

Para una tasa j se calcula `v = (I - P_Kotros) kj` y, en una región G, `z = sum(v' Y[:,G]) / (sigma ||v|| sqrt(|G|))`. El umbral normal con corrección por el número de comparaciones es un **filtro condicional**: como las tasas se estiman, no constituye un p-valor calibrado ni una garantía formal de control de falsos positivos. Las alternativas filtradas compiten con las no filtradas sobre datos reservados. La desviación de la diferencia de SSE predictiva es `2 ||pred1-pred2|| / sigma`, condicional a los ajustes previos; seleccionar el mejor entre varios candidatos tampoco convierte esta regla en un intervalo simultáneo.

Se asume ruido gaussiano independiente, sigma conocido y espectros de absorción correctamente faseados. La correlación instrumental del ruido, baseline, deriva de frecuencia, intercambio y difusión no monoexponencial requieren validación adicional. Las ventanas se separan por huecos en las frecuencias seleccionadas; ventanas mal definidas pueden unir resonancias distintas. Las especies cercanas o con espectros proporcionales pueden no identificarse. Una señal débil no detectada se considera **no resuelta**, no químicamente ausente.

El banco anterior tiene distribuciones con anchura: es útil como prueba de transferencia, pero no es ground truth de moléculas estrictamente discretas. Se añadieron 48 problemas independientes, con escenarios discretos, minoritarios del 4%, tasas próximas, espectros proporcionales, controles nulos y distribuciones anchas. Los resultados por escenario y las regresiones aparecen en `INFORME.md`.

## RAI-Net

Se conserva el controlador físico y la verificación de descenso y brecha primal-dual. La modificación aprende a ordenar acciones usando mejora de predicción en gradientes internos reservados, en lugar de mejora del objetivo de entrenamiento. Se generan 80 problemas DOSY con frecuencias correlacionadas, separados por problema completo para entrenamiento y validación de épocas. La red tiene dos capas ocultas de 48 unidades y 400 épocas con restauración de la mejor época de validación.

El checkpoint, las trazas, las semillas y el historial se guardan en `training/`. La red es una política para discretización adaptativa, **no un clasificador del número de especies ni el selector RAI-S**. Su utilidad se decide con la comparación de test, no con el descenso de la pérdida de entrenamiento. El módulo original y el reentrenado se evalúan con idéntico controlador y presupuesto de inferencia.

## Uso desde MATLAB

```matlab
addpath('SOURCE_ROOT/dosy_support_v2_20260925');
r = DOSY_RAI_DOME_S(Y, b, ppm, sigma, Method="RAI", Output="resultado_rai.mat");
r = DOSY_RAI_DOME_S(Y, b, ppm, sigma, Method="DOME", Output="resultado_dome.mat");
```

`Y`: gradientes × frecuencias; `b`: s/m²; `ppm`: coordenadas del espectro; `sigma`: desviación del ruido en unidades de Y. No introducir el gradiente G como b sin calcular el factor de difusión. Puede proporcionarse `Mask=mascara_logica`. Sin máscara, se selecciona señal usando los primeros gradientes de descubrimiento y se amplía dos puntos. No se debe tomar el módulo de espectros con ruido gaussiano y conservar esta misma suposición de ruido.

`r.r_auto`: número de componentes elegido automáticamente; `r.D`: m²/s; `r.A`: componentes × frecuencias seleccionadas; `r.X`: 256 × frecuencias seleccionadas, masas por bin; `r.support`: soporte retenido. `r.diagnostics.component_selection` documenta la decisión automática. `r.diagnostics` también contiene candidatos, gradientes reservados, precisión local, frontera y discrepancia del modelo atómico. El intervalo implementado en esta versión es `[0.1,15] × 10^-9 m²/s`; hasta cuatro componentes. Cambiarlo exige un protocolo y validación apropiados, no ampliar la figura únicamente.

Python se configura explícitamente con `Python="ruta/python.exe"`. Dependencias de inversión: NumPy y SciPy. PyTorch es necesario solo para entrenamiento/inferencia de RAI-Net. Matplotlib se usa para figuras.

## Reproducción

Desde esta carpeta, con el Python registrado en `environment.json`:

```text
python test_support.py
python run_benchmark.py --partition development --count 24
python train_net.py --problems 80 --workers 2
python run_benchmark.py --partition test --count 48
python run_benchmark.py --partition archive
python evaluate_net.py --partition test --count 24
python evaluate_net.py --partition archive
python render.py
python report.py
```

Los resultados existentes se reutilizan. Para una repetición independiente, copiar código y `vendor/` a otra carpeta vacía, conservar el benchmark anterior en la ruta relativa indicada en `io_utils.py`, y ejecutar de nuevo. No borrar resultados para ocultar ensayos previos. `protocol_frozen.json` registra la configuración y hashes antes del test; `results/development_initial/` conserva el ensayo de desarrollo inicial. Los 24 casos anteriores y el checkpoint original se leen del benchmark congelado adyacente. No se modifica ningún experimento RMN ni el ajuste MF.

`verify_matlab` comprueba el NNLS de componentes contra `lsqnonneg` de MATLAB y ejecuta la interfaz completa. La comparación numérica se hace sobre las predicciones, porque amplitudes individuales pueden no ser únicas en sistemas degenerados. [Referencia de NNLS y condiciones KKT en SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html).

Esta entrega no demuestra aún recuperación química en muestras reales, ni que MF sea perfecto en polímeros, ni una ventaja universal de RAI o DOME. Permite reproducir una mejora concreta y sus límites medidos.
