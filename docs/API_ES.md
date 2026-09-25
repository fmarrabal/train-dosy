# Contrato de la API (v1)

El nombre público del algoritmo es **TRAIn-MF**. Se conserva `MF-AUTO` como identificador de API/CLI y `DOSY_MF_Auto` como función MATLAB. Ambos invocan la misma referencia numérica.
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
| sigma | Desviación estándar positiva del ruido homogéneo conocido, en unidades de Y. |
| method | RAI-S, DOME-S o MF-AUTO. TRAIn-MF necesita MATLAB configurado y con licencia. |
| mask | Máscara booleana opcional; true incluye una frecuencia. Las excluidas no se estiman. |
| diffusion_bounds | Intervalo positivo creciente en m²/s; defecto [1e-10, 1.5e-8]. |
| bins | De 256 a 2048. En RAI-S/DOME-S afecta a la representación; las tasas se optimizan fuera de la malla. En TRAIn-MF es la malla real de optimización. |
| max_components | Límite de búsqueda 1–4 para los métodos atómicos, defecto 4. No fija el resultado. TRAIn-MF conserva el límite de 4. |

Se rechazan campos adicionales, valores no finitos, coordenadas repetidas, dimensiones incorrectas, máscaras vacías, menos de 12 o más de 256 gradientes, más de 8192 frecuencias y más de 524288 observaciones. El cuerpo HTTP no puede superar 16 MiB.

Sin mask se usan hasta seis gradientes bajos excluyendo los reservados para validación interna. Se selecciona la media > 4 sigma/sqrt(n) y se amplía dos posiciones espectrales. La dilatación presupone columnas contiguas en el eje adquirido. Para disolventes, ejes irregulares, fase deficiente o señales anchas, aportar una máscara apropiada. La ausencia de selección no demuestra ausencia física.

La máscara explícita debe ser previa o calcularse sin datos de prueba externa y preferiblemente sin validación interna. El benchmark histórico mantiene su máscara calculada sobre entrenamiento externo: contiene información de validación interna y su criterio se describe como heurístico.

## Resultados
En los métodos atómicos, D contiene tasas fuera de malla; A tiene r por n_seleccionadas. X contiene **masa por bin**, con bins por n_seleccionadas. logD_edges son bordes en logaritmo natural de D en SI; X/diff(logD_edges) es densidad. prediction usa las tasas exactas, conserva el orden de gradientes y solo incluye las frecuencias seleccionadas.

TRAIn-MF devuelve D_grid, X, S y A, con X=S A. Su rango cuenta factores predictivos compartidos, no especies químicas. Hay que comprobar success, status y kkt antes de interpretar un resultado.

Ambos devuelven mask, selected_frequency_indices (índices desde cero), ppm, selected_rank, selection_mode, search_limit, search_limit_reached, units y software_version. El rango cero es válido en ventanas de ruido suministradas explícitamente; si el descubrimiento automático no detecta señal se devuelve 422.

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
Arrancar después el servidor o usar la CLI con --method MF-AUTO. Las rutas las fija el operador, nunca la petición. Cada ajuste TRAIn-MF inicia MATLAB en modo batch. MATLAB y su lsqnonneg no se redistribuyen. Ejemplo directo: matlab/example_mf.m.

