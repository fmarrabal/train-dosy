# Contrato de la API (v1)

El nombre público del algoritmo es **TRAIn-MF**. El software 0.2.1 utiliza el protocolo `signed-v2.2` para `TRAIn-MF`; `MF-AUTO` conserva el solver restringido del artículo (`DOSY_MF_Auto`). Son protocolos numéricos explícitamente distintos dentro de la evolución del mismo método. Véase la [guía de corrección y migración](TRAIN_MF_SIGNED_ES.md).
La inversión se realiza conjuntamente sobre las frecuencias seleccionadas. La API no acepta especie química, número verdadero de componentes, valor objetivo de difusión ni archivo de verdad de referencia.

## Arranque y llamada
Instalar el extra api desde la raíz del repositorio:
~~~sh
python -m uvicorn train_dosy.api:app --host 127.0.0.1 --port 8765
~~~
Documentación interactiva: http://127.0.0.1:8765/docs . Esquema: /openapi.json . Estado: GET /health . Cálculo: POST /v1/fit con application/json.
~~~sh
curl -X POST http://127.0.0.1:8765/v1/fit -H "Content-Type: application/json" --data-binary @examples/three_components/input.json -o resultado.json
~~~
En Windows PowerShell, usar curl.exe.

## Campos de entrada
| Campo | Significado |
|---|---|
| Y | Matriz real con signo: gradientes por frecuencias. Debe estar faseada; no recortar el ruido negativo. |
| b | Factores físicos de atenuación distintos y no negativos, en s/m². No son simplemente G². |
| ppm | Desplazamientos químicos distintos, uno por columna, ascendentes o descendentes. |
| sigma | Escalar positivo obligatorio: desviación estándar obtenida independientemente del residuo ajustado, en unidades de Y. El control de TRAIn-MF presupone ruido gaussiano iid homogéneo. |
| method | RAI-S, DOME-S, MF-AUTO o TRAIn-MF. Los dos MF necesitan MATLAB configurado y con licencia. |
| mask | Máscara booleana opcional; true incluye una frecuencia. Las excluidas no se estiman. |
| diffusion_bounds | Intervalo positivo creciente en m²/s; defecto [1e-10, 1.5e-8]. |
| bins | De 256 a 2048. En RAI-S/DOME-S afecta a la representación; las tasas se optimizan fuera de la malla. En TRAIn-MF es la malla real de optimización. |
| max_components | Límite de búsqueda 1–4, defecto 4. No fija el resultado. Solo MF-AUTO exige el límite fijo de 4. |

Se rechazan campos adicionales, valores no finitos, coordenadas repetidas, dimensiones incorrectas, máscaras vacías, menos de 12 o más de 256 gradientes, más de 8192 frecuencias y más de 524288 observaciones. El cuerpo HTTP no puede superar 16 MiB.

Sin mask se usan hasta seis gradientes bajos excluyendo los reservados para validación interna. Se selecciona la media > 4 sigma/sqrt(n) y se amplía dos posiciones espectrales. La dilatación presupone columnas contiguas en el eje adquirido. Para disolventes, ejes irregulares, fase deficiente o señales anchas, aportar una máscara apropiada. La ausencia de selección no demuestra ausencia física.

La máscara explícita debe ser previa o calcularse sin datos de prueba externa y preferiblemente sin validación interna. El benchmark histórico mantiene su máscara calculada sobre entrenamiento externo: contiene información de validación interna y su criterio se describe como heurístico.

## Resultados
En los métodos atómicos, D contiene tasas fuera de malla; A tiene r por n_seleccionadas. X contiene **masa por bin**, con bins por n_seleccionadas. logD_edges son bordes en logaritmo natural de D en SI; X/diff(logD_edges) es densidad. prediction usa las tasas exactas, conserva el orden de gradientes y solo incluye las frecuencias seleccionadas.

Ambos MF devuelven D_grid, X, S y A, con X=S A. Su rango cuenta factores predictivos compartidos, no especies químicas. MF-AUTO informa del residuo KKT. TRAIn-MF informa del protocolo `signed-v2.2`; su campo kkt es null, porque no calcula estacionariedad KKT conjunta. Sus diagnósticos separan `numerical_converged`, `model_compatible`, `selected_rank`, `active_rank`, `rank_selection_resolved`, `boundary_hit` y `component_boundary_hit`. Los factores exactamente inactivos se eliminan de S/A, cuya dimensión de factores es `active_rank`; se mantiene `selected_rank` nominal y una discrepancia deja el resultado sin resolver. `candidate_numerical_converged` registra la parada separadamente: la selección predictiva puede usar candidatos finitos y activos que no alcanzaron su objetivo numérico. `success` exige superar todos los controles finales. La convergencia numérica sola no aprueba una D sesgada o incompatible.

La API exige sigma y la pasa al detector, a la evidencia en validación contra el modelo nulo y al control independiente de compatibilidad residual de TRAIn-MF, con un umbral superior chi-cuadrado del 99% por defecto para N observaciones y errores gaussianos iid entre todas las celdas. Las señales espectrales subyacentes sí pueden estar correlacionadas: la hipótesis se refiere al ruido de medida. Sigma no sustituye la referencia numérica NNLS. El control de compatibilidad es conservador: no produce un valor p calibrado tras seleccionar el modelo ni demuestra identificabilidad. MATLAB directo permite omitir sigma: `model_compatible` queda NaN (null en JSON) y `success=false`. No convertir esa compatibilidad desconocida en verdadera. Con rango fijo en MATLAB, `rank_selection_applicable=false`, `rank_selection_resolved=false` y `success=false`. Si cambian las unidades de intensidad, escalar sigma junto con Y.

Ambos devuelven mask, selected_frequency_indices (índices desde cero), ppm, selected_rank, selection_mode, search_limit, search_limit_reached, units y software_version. TRAIn-MF incluye r=0 en la selección automática. Un modelo nulo respaldado tiene X=0, S de dimensiones bins por 0, A de 0 por n_seleccionadas, predicción cero y media/moda de D indefinidas. JSON representa S como una lista de filas vacías y A como []; reconstruir la segunda dimensión de A con el número de frecuencias seleccionadas. `no_signal_supported` puede tener `success=true`: se acepta el modelo cero según esos controles, sin identificar un componente. Si el descubrimiento automático no detecta señal se devuelve 422.

Los diagnósticos atómicos incluyen pérdidas de candidatos, soporte, estacionariedad, límites de difusión, precisión deficiente e incompatibilidad con el modelo atómico. No son intervalos de confianza química calibrados. Los diagnósticos no finitos se representan como null.

## Errores y despliegue
200: cálculo terminado, revisar success. 422: entrada inválida o sin señal detectada. 429: ya existe un cálculo activo. 503: no hay candidato verificado o falta MATLAB. Los fallos inesperados producen 500 sin traza interna en la respuesta. Se permite un cálculo simultáneo; no se aceptan rutas del servidor, órdenes shell ni direcciones remotas en las solicitudes.

Es una API de investigación probada en local; no existe un servicio alojado públicamente. Un despliegue en Internet requiere autenticación, proxy con límites, aislamiento y revisión operativa. Cancelar el cliente no garantiza parar el ajuste Python. El subproceso MATLAB tiene un límite de 600 segundos.

## Backend TRAIn-MF nativo opcional
Instalar MATLAB y Optimization Toolbox; configurar en el proceso del servidor:
~~~powershell
$env:TRAIN_DOSY_MATLAB = 'C:\Program Files\MATLAB\R2026a\bin\matlab.exe'
$env:TRAIN_DOSY_MATLAB_FUNCTIONS = (Resolve-Path matlab).Path
~~~
Arrancar después el servidor o usar la CLI con `--method TRAIn-MF` (revisado) o `--method MF-AUTO` (congelado). Las rutas las fija el operador, nunca la petición. Cada ajuste inicia MATLAB en modo batch. MATLAB y su lsqnonneg no se redistribuyen. La [guía de corrección](TRAIN_MF_SIGNED_ES.md) incluye el uso MATLAB revisado; `matlab/example_mf.m` conserva el ejemplo del artículo.
