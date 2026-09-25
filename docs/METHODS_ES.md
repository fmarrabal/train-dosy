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

