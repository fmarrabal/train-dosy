# Corrección y diagnósticos de TRAIn-MF con datos con signo (software 0.2.1)

[English](TRAIN_MF_SIGNED_EN.md)

La revisión **signed-v2.2** utiliza `lsqnonneg` del MATLAB instalado. El port anterior signed-v2.1 eliminó la resta del mínimo de las atenuaciones y el recorte de los residuos de componentes, y seleccionó el rango mediante predicción en gradientes reservados. La versión 0.2.1 añade un modelo nulo, comprueba los límites de la rejilla y el rango activo, y separa la convergencia numérica de la compatibilidad con ruido aportado de forma independiente. No recalibra D ni desplaza distribuciones hacia masas esperadas.

El nuevo identificador de API/CLI es **TRAIn-MF**. **MF-AUTO** conserva el solver restringido utilizado en el artículo. Se mantienen intactos `matlab/legacy/TRAIn_DOSY_MFV31.m`, los núcleos congelados, sus resultados y el manuscrito enviado. Son implementaciones sucesivas del mismo método multifrecuencia, con protocolos de optimización y selección diferentes.

## Formulación

Para observaciones reales Y, con gradientes en filas y frecuencias espectrales en columnas, K(i,l)=exp(-b(i)D(l)). Se requieren b en s/m², D en m²/s y al menos 256 nodos crecientes. El objetivo es

    F(S,A) = (1/2) ||K S A - Y||_F²,
    S >= 0, sum(S(:,k)) = 1, A >= 0.

X=S A contiene masa de señal por nodo, no densidad. Un perfil compartido puede ser ancho o multimodal; el rango no equivale al número de moléculas.

Se aplica una escala RMS común dentro de cada partición de entrenamiento y se recupera al final; los datos de validación no influyen en las escalas ni tolerancias del ajuste de candidatos. No se resta el mínimo de cada curva, no se rectifican observaciones negativas, no se normalizan frecuencias por separado, no se suavizan amplitudes después de ajustarlas ni se adivinan las unidades de b. Una señal que persiste al último gradiente no es necesariamente fondo.

Para actualizar el perfil k se calcula, conservando el signo,

    R_k = Y - sum(j != k) (K S(:,j)) A(j,:),
    y_k = R_k A(k,:)' / ||A(k,:)||².

TRAIn resuelve el problema con el kernel físico mediante h=eta², productos Gauss–Newton y región de confianza de Steihaug. Se normaliza el perfil candidato y se reajustan todas las amplitudes con NNLS de MATLAB. Solo se acepta si no aumenta el residuo global. No se añaden penalizaciones lambdaS/lambdaA a la matriz ni a la referencia residual.

La parada de TRAIn compara el residuo con `term_factor * ||K h_NNLS-y_k||`, con term_factor=1,05 por defecto, intervalo 1,02–1,05 y tolerancia aritmética `100*eps*||y_k||`. El residuo NNLS es una **referencia numérica operativa**, no una medida independiente de ruido. Esta regularización por parada no certifica el mínimo global ni las condiciones KKT conjuntas.

Frente a signed-v2 de DiffAtOnce, esta revisión conserva las colas pequeñas de la distribución, verifica las semillas y permite 2000 iteraciones internas por defecto. Signed-v2.2 también escala cada problema interno de forma homogénea e impone un mínimo positivo relativo en la inicialización: tanto los ecos nulos como los positivos subnormales podían anularla al elevar las variables al cuadrado. Trata explícitamente una solución NNLS nula. Por estas diferencias documentadas no se promete identidad bin a bin entre C# y MATLAB.

## Selección automática y diagnósticos

La selección automática incluye **r=0**, cuya predicción es cero, junto a los rangos de 1 a r_max; exige al menos ocho gradientes, incluso con r_max=1. Los rangos positivos se ajustan sin usar las filas de validación. Las predicciones elegibles deben ser finitas y conservar factores activos. Con sigma conocida se compara cada predicción obtenida del entrenamiento con cero en validación independiente, mediante una proyección gaussiana y un umbral de Bonferroni entre rangos positivos. Si un candidato elegible aporta evidencia de señal, se excluye cero antes de comparar los rangos positivos respaldados. Se prefiere el menor dentro de un error estándar emparejado frente al mejor y se reajusta con todas las filas. Sin sigma, cero compite directamente en la regla de un error estándar. Esa comparación es heurística: no es un intervalo de confianza ni garantiza el rango globalmente óptimo.

`candidate_numerical_converged` registra la parada numérica separadamente de la elegibilidad. Un candidato finito que se estancó puede influir en el rango predictivo: `rank_selection_resolved` solo indica que la regla tomó una decisión cuyo número de factores sigue activo tras el reajuste. No afirma convergencia del optimizador, mínimo global ni identificación química. Los factores exactamente inactivos se eliminan de S/A; `selected_rank` mantiene el rango nominal y el resultado no se resuelve si difiere de `active_rank`.

Los diagnósticos responden a preguntas diferentes:

| Campo | Interpretación |
|---|---|
| `numerical_converged` | Se cumplen los criterios numéricos aplicables de residuo y subproblemas. No valida D. |
| `model_compatible` | Se supera la comprobación con ruido independiente descrita abajo. NaN en MATLAB o null en JSON indica que falta sigma y la compatibilidad es desconocida. |
| `selected_rank`, `active_rank` | Número de factores seleccionado y número activo tras el ajuste. Ninguno cuenta especies químicas. |
| `rank_selection_resolved` | La regla automática tomó una decisión y coincide el número activo final. Con rango fijo, `rank_selection_applicable=false` y `rank_selection_resolved=false`. |
| `boundary_hit`, `component_boundary_hit` | Una distribución espectral o un perfil compartido tiene la moda en un extremo o acumula señal cerca de los límites. Revisar y, si tiene justificación científica, repetir con un intervalo más amplio elegido independientemente. |
| `success` | Se cumplen conjuntamente los controles numéricos, de compatibilidad, rango y límites aplicables. No establece unicidad física ni identidad química. |

Para un ajuste no nulo, `success` exige convergencia numérica, compatibilidad conocida, rango automático resuelto y ausencia de avisos de límites en columnas/perfiles. En caso contrario se requiere revisión; por ello, un rango fijo tiene `success=false` aunque ajuste bien. Por defecto se avisa cuando la moda está en un extremo o al menos el 10% de la masa cae en el 2% exterior de cada extremo del intervalo log-D, sumando ambos. Son umbrales ajustables de aviso, no una corrección del sesgo. Un rango predictivo y una D media correctos pueden coexistir con `numerical_converged=false`, porque el NNLS independiente dispone de más libertad para ajustar ruido. No se añaden factores solo para aprobar esa comprobación. Un resultado nulo compatible puede tener `success=true` y `no_signal_supported`: permite conservar el modelo cero según estos controles, sin identificar D. X es cero, S tiene dimensiones nD por 0, A tiene 0 por n_seleccionadas y la media/moda de D son indefinidas.

En MATLAB directo, `sigma` es opcional y debe ser una **desviación estándar escalar positiva obtenida de forma independiente del residuo ajustado**, en las mismas unidades que Y. Se presupone ruido gaussiano iid entre todas las celdas, incluidas las frecuencias. La independencia se exige a los errores, no a las señales: los perfiles espectrales compartidos y correlacionados siguen siendo el objetivo del MF. Se compara el residuo cuadrático/sigma² con el umbral superior chi-cuadrado del 99% por defecto usando N observaciones suministradas. Es un control conservador de falta de ajuste: no estima grados de libertad efectivos, no produce un valor p calibrado tras seleccionar el modelo ni demuestra identificabilidad. El ruido correlacionado, heterogéneo o contaminado por fondo incumple esas hipótesis. Sin sigma se puede calcular, pero `model_compatible` queda desconocido y `success=false`; el residuo NNLS no sustituye una medida independiente de ruido. Si se escala Y, debe escalarse sigma por el mismo factor.

En MATLAB se reserva aproximadamente un cuarto de los gradientes interiores, de forma determinista; también se acepta `validation_rows` lógico. La máscara espectral debe construirse sin esas filas. La API reserva las mismas filas que excluye al detectar señal. La API exige sigma y la pasa tanto al detector como al control independiente de compatibilidad; la referencia numérica NNLS sigue separada. La evaluación independiente necesita gradientes externos.

## Uso

```matlab
addpath('matlab');
D = logspace(log10(.02e-9),log10(5e-9),256)';
% sigmaIndependiente procede de una medida de ruido independiente y apropiada.
[X,D,info] = TRAIn_DOSY_MF_Signed(Y,b,D, ...
    struct('r_max',4,'sigma',sigmaIndependiente));
% Entrada compatible con el formato de rejilla anterior, ya corregida:
[X,D_nano,info] = TRAIn_DOSY_MF(Yfull,b,[.02 5 256], ...
    struct('signal_mask',signalMask,'n_components','auto','r_max',4, ...
           'sigma',sigmaIndependiente));
```

La máscara lógica selecciona regiones completas con señal. Las columnas excluidas son NaN, porque no se estiman. `info.A` y los diagnósticos siguen el orden de `info.process_idx`. `auto_r=false` exige un `n_components` numérico explícito; si falta, produce `TRAInMF:FixedRank` en lugar de activar la selección automática. Se rechazan opciones antiguas incompatibles de normalización, suavizado, unidades automáticas o penalizaciones. La versión V3.1 histórica sigue disponible para reproducir resultados anteriores.

Tras configurar el MATLAB con licencia según [la API](API_ES.md):

```sh
train-dosy examples/one_component/input.json resultado-revisado.json --method TRAIn-MF
```

Python y el cliente C# llaman al mismo solver MATLAB. `max_components`, de 1 a 4, limita la búsqueda; no impone el rango verdadero. HTTP 200 solo confirma que terminó el cálculo. La respuesta incluye `protocol`, `diagnostics`, S y A; `kkt` es null, porque esta implementación no calcula un certificado KKT conjunto.

## Comprobación reproducible

```matlab
addpath('matlab/tests');
report = test_train_mf_signed('verification/train_mf_v22_recovery.json');
guards = test_train_mf_guards('verification/train_mf_v22_guards.json');
stress = test_train_mf_v22_stress('verification/train_mf_v22_stress.json');
```

La batería original permanece intacta: 1, 2 y 3 perfiles anchos compartidos, correlación entre frecuencias y solapamiento, escalado, permutación, colas con signo, máscaras y sesgo por restar el mínimo. La nueva batería añade sigma sintética conocida independientemente, ruido puro, datos nulos, D verdadera por debajo/encima de la rejilla, sigma ausente/inválida, opciones de rango fijo y un eco positivo de 1e-320. Registra la convergencia separada de la precisión y comprueba que no se aprueben modelos incompatibles; no fuerza la convergencia de todo problema exacto o sin ruido. La verdad se usa solo para evaluar.

La batería de estrés registra 20 realizaciones independientes de ruido sintético, factores de intensidad 1e-150 y 1e150 escalando sigma a la vez, aislamiento del entrenamiento al cambiar valores de validación, y comprobaciones algebraicas y por diferencias finitas del gradiente condicional y los productos Gauss–Newton. La ejecución signed-v2.2 guardada seleccionó cero en las 20 realizaciones nulas; esa muestra finita no calibra una tasa de falsos positivos experimental. Los nombres versionados conservan las verificaciones anteriores.

Son regresiones sintéticas, no validación experimental nueva. Recuperar bien la media o la atenuación no garantiza recuperar la moda o la anchura. No se consideran resueltas las limitaciones del mapa completo observadas en DiffAtOnce.
