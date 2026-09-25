# Contrato matemático y límites

## ¿Por qué TRAIn-MF?
**MF significa multifrecuencia**, de acuerdo con la cabecera original de `TRAIn_DOSY_MFV31`. Matemáticamente se implementa mediante factorización matricial no negativa, `Y ≈ K S A`: las columnas de `S` son perfiles de difusión compartidos y las filas de `A` describen su intensidad en las distintas frecuencias. Sigue siendo multifrecuencia si el rango seleccionado es uno. Un perfil puede ser ancho o multimodal; el rango no equivale al número de sustancias.

**TRAIn-MF: extensión multifrecuencia de TRAIn mediante factorización matricial no negativa** es el nombre del algoritmo. La referencia actual con restricciones sustituye la iteración TRAIn interna original y no es numéricamente idéntica a V3.1. Se conservan los nombres de las funciones, los bytes del original, los núcleos numéricos y los resultados guardados. Consulta la [bibliografía ampliada](BIBLIOGRAPHY_ES.md).

El modelo es Y(i,l)=integral exp(-b_i D) d mu_l(D)+ruido. D es difusión. Convertirlo a peso molecular requiere una calibración justificada aparte.

## TRAIn-MF distribucional
Y≈K S A, K(i,j)=exp(-b_i D_j), S≥0 con columnas de masa unidad y A≥0. Se minimiza
F = ||K S A - Y/s||_F²/(2p) + lambda_S ||L S||_F²/2 + lambda_A ||A||_F²/(2p).
s es el RMS de Y; L actúa sobre la densidad en log-difusión e incluye cuadratura. A se devuelve en unidades de señal. DOSY_MF_Auto usa por defecto lambda_S=lambda_A=1e-8, 256 bins, dos inicios y tolerancia KKT 1e-6. La disminución escalar no garantiza factores únicos ni óptimo global. La selección automática limita la búsqueda por las direcciones singulares resueltas sobre ruido y compara error predictivo reservado. Ese rango no es una cota superior del número de especies. Una convergencia insuficiente o un límite de búsqueda insuficiente se notifican como no resueltos.

## RAI-S y DOME-S discretos
Y≈exp(-b D) A, con tasas positivas continuas y amplitudes no negativas. Las propuestas difieren; el selector final es común. Las amplitudes se calculan por NNLS enumerando soportes activos, con hasta cuatro componentes. El ajuste reducido usa la derivada completa de proyección variable sobre conjuntos activos estables. La puntuación de núcleos residualizados se agrega en ventanas espectrales separadas. Compiten candidatos con y sin selección de soporte sobre gradientes reservados; entre predicciones comparables se prefieren menos tasas y luego menos parámetros de soporte. No se elimina por un porcentaje fijo de la altura máxima.

Valores: max_components=4, alpha=.01, validation_stride=4, selection_se=1, max_nfev=350. Se comprueban KKT de amplitudes y estacionariedad de tasas; se devuelven alertas de precisión, límites y ajuste atómico inadecuado. La puntuación con tasas y máscaras estimadas es heurística; no es una garantía de detección química.

## Formulaciones específicas de manuscript-v4

La [sección editable de métodos](../paper/sections/methods.tex), el [apéndice matemático](../paper/sections/method_details.tex) y el [mapa de ecuaciones y 27 archivos fuente](../paper/evidence/method_formulations.json) especifican operadores, incógnitas, restricciones, objetivos, actualizaciones, regularización, selección y límites numéricos. El Algoritmo 1 usa un entorno real `algorithm`/`algpseudocode`, con líneas numeradas, entradas, salidas, bucles, comprobaciones de convergencia, selección y reajuste final; remite a las ecuaciones correspondientes.

n es el número de adquisiciones; p, las frecuencias con señal; q, las celdas de difusión; r, los factores compartidos. El operador de una base integra exp(-b D) contra funciones positivas de integral unidad: sus coeficientes son masas. Las tasas atómicas permanecen continuas; asignarlas a bins para dibujar no convierte el ajuste en una inversión sobre una malla.

| Método | Inversión y selector específicos |
|---|---|
| TRAIn adjunto | h=eta²; minimiza ||K h-y||². Matriz Gauss-Newton B=8 diag(eta) KᵀK diag(eta), paso de región de confianza mediante CG truncado y cociente de reducción real/predicha. Para respecto al residual de NNLS. El núcleo no incluye una penalización explícita de Tikhonov. |
| RAI atómico nativo | Elimina A≥0 de ||(K(d)A-Y)/sigma||²/2 por NNLS. L-BFGS-B acotado en log d, diez inicios y nacimientos basados en residuales. Selector RSS/sigma²+r(p+1)log(np). |
| RAI de densidad | Masas C≥0; pérdida de datos/(2n) + lambda_s||R C||²/2 + eta tr(CᵀH C)/2 + gamma tr(H C L_grafo Cᵀ)/2. R actúa sobre diferencias de densidad; H=diag(1/anchura de bin). La cresta positiva da unicidad con malla fija. La validación decide los refinamientos. |
| RAI Haar adaptativo | Masas de hoja normalizadas U≥0; masas físicas P_arbol U. Pérdida/(2n)+eta||U||²/2+lambda sqrt(p) suma de normas de filas de detalle Haar. ADMM con proyección positiva y contracción por grupos; brecha primal-dual en cada árbol. La validación compara trayectorias de regularización completas. |
| RAI-Net original / revisado | Red de dos capas tanh que ordena acciones de árbol con 14 características definidas. Pérdida Smooth-L1 sobre ganancias transformadas: descenso del objetivo de ajuste en la original; mejora predictiva en validación interna en la revisada. Conserva objetivo Haar y comprobaciones numéricas; no clasifica r ni predice directamente densidades. |
| DOME nativo | Mismo residual atómico, NNLS por subconjuntos activos y región de confianza con Jacobiano residual completo. Curvatura, contrastes espectrales y divisiones/intercambios de momentos generan candidatos. Un criterio corregido de número de parámetros selecciona el orden. |
| RAI-S / DOME-S | Mantienen distintas propuestas; criban soportes espectrales, refinan cada candidato y comparan predicción reservada. Prefieren menos tasas y después menos amplitudes permitidas dentro de la tolerancia pareada. El Algoritmo 1 fija el orden exacto. |
| TRAIn-MF | Perfiles S positivos de masa unidad y espectros A≥0; objetivo regularizado y NNLS real de MATLAB. QP conjunto sobre simplex para S y SQP perfilado cuando hace falta. Evidencia de rango limita la búsqueda; el error estándar empírico de pérdidas pareadas por adquisición selecciona factores predictivos. |
| DOME con promedio | Tasas nativas fijas; NNLS en cada subconjunto espectral y pesos suaves con entropía, puntuación RSS/sigma²+6 tamaño del subconjunto y temperatura 0.5. No equivale a DOME-S. |
| RAI-FLEX | Y≈[K(d),G C]A; perfiles spline C positivos de masa unidad y penalización de segunda derivada lambda||L₂C||²/2. NNLS perfilado, optimización log-tasa/softmax y refinamiento sobre simplex. Selección heurística de átomos/perfiles por dimensión efectiva local. |
| Partial-C | C=s a+V≥0 con un perfil compartido de masa unidad; penaliza la primera derivada de la densidad total y la norma L² de la privada. NNLS condicional conjunto para (a,V) y SLSQP exterior. Rango compartido fijado a uno en este comparador. |
| Adaptaciones CIRCE v1 / v2 | Cuadráticas convexas positivas con suavidad y grafo espectral. V2 impone C=T Z, Z≥0, mediante un decodificador positivo fijo que conserva masa y suavidad escalada por columna. FISTA o NNLS cíclico estricto; después comprueba cono residual y cotas de masa. No se atribuyen inferencia CIRCE-Net ni cotas de medidas alternativas a estas adaptaciones. |

Ejecuta `python paper/scripts/check_formulations.py` para 20 comprobaciones algebraicas deterministas: gradientes, identidad QR/Kronecker, brecha Haar factible, momentos de DOME y conservación de masa del decodificador. No repite ajustes ni demuestra recuperación estadística. Registra los resultados en `verification/formulation_checks.json`.

### Diferencias que afectan a la interpretación

- `lambdaSparse` de TRAIn-MFV31 añade sqrt(lambda) diag(sqrt(w)) al diseño: produce una penalización cuadrática ponderada, no L1 explícita. El recorte de pseudocurvas y las modificaciones de suavidad/pesos no equivalen al objetivo fijo actual de TRAIn-MF.
- El error estándar empírico pareado de TRAIn-MF difiere de la tolerancia de distancia predictiva gaussiana de RAI-S/DOME-S. Ninguno constituye una prueba calibrada del número de especies tras máscaras y selección adaptativas.
- Un subproblema convexo puede tener mínimo único sin que lo tengan el árbol, la malla, los factores o las especies. La parametrización cuadrática puede tener gradiente cero sin cumplir KKT de NNLS.
- El alcance ejecutable de la API sigue siendo el del apartado de reproducción. Formular variantes históricas no incorpora su implementación completa ni reentrenamiento al paquete. Se mantienen el software numérico, estimaciones guardadas, métricas y figuras.

## Identificación
Un conjunto finito de observaciones Laplace no identifica una medida positiva arbitraria. Pueden fallar las especies próximas, débiles, con espectros proporcionales, las fases/bases incorrectas y el ruido o intervalo mal especificados. Más bins no aportan información experimental. Acertar el rango es distinto de recuperar todas las tasas y de identificar sustancias.

## Alcance reproducible
- examples: tres ejemplos de desarrollo regenerables mediante scripts/make_examples.py.
- benchmarks/run_atomic.py: 48 problemas y cuatro métodos atómicos; 36 discretos con señal, seis nulos y seis controles anchos. Conserva la máscara histórica calculada con todo el entrenamiento externo, incluida la validación interna.
- benchmarks/frozen/metrics.csv: métricas previas por problema, incluidos archivos históricos y resultados neuronales exploratorios; no se han cambiado.
- paper/scripts/prepare_evidence.py: tablas y figuras desde entradas sintéticas guardadas, sin reajuste.
- MATLAB y backend opcional: cálculo TRAIn-MF nativo real, con licencia. La figura TRAIn-MF del manuscrito conserva el resultado anterior y su hash.
- C#: cliente y CLI tipados para la API; no es otra implementación independiente ni otra validación científica.

El entrenamiento neuronal y el rastreo completo de métodos históricos no forman parte de esta publicación de código seleccionada. Se conservan sus métricas citadas, sin prometer que este paquete reentrene esos modelos.

