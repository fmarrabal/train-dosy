# Corrección de TRAIn-MF con datos con signo (software 0.2.0)

[English](TRAIN_MF_SIGNED_EN.md)

La revisión **signed-v2.1** traslada a MATLAB la corrección desarrollada en DiffAtOnce y utiliza `lsqnonneg` del MATLAB instalado. Elimina la resta del mínimo de las atenuaciones y el recorte de los residuos de componentes. No recalibra D ni desplaza distribuciones hacia masas esperadas. Además, selecciona el rango mediante predicción en gradientes reservados.

El nuevo identificador de API/CLI es **TRAIn-MF**. **MF-AUTO** conserva el solver restringido utilizado en el artículo. Se mantienen intactos `matlab/legacy/TRAIn_DOSY_MFV31.m`, los núcleos congelados, sus resultados y el manuscrito enviado. Son implementaciones sucesivas del mismo método multifrecuencia, con protocolos de optimización y selección diferentes.

## Formulación

Para observaciones reales Y, con gradientes en filas y frecuencias espectrales en columnas, K(i,l)=exp(-b(i)D(l)). Se requieren b en s/m², D en m²/s y al menos 256 nodos crecientes. El objetivo es

    F(S,A) = (1/2) ||K S A - Y||_F²,
    S >= 0, sum(S(:,k)) = 1, A >= 0.

X=S A contiene masa de señal por nodo, no densidad. Un perfil compartido puede ser ancho o multimodal; el rango no equivale al número de moléculas.

Se aplica una única escala RMS global y se recupera al final. No se resta el mínimo de cada curva, no se rectifican observaciones negativas, no se normalizan frecuencias por separado, no se suavizan amplitudes después de ajustarlas ni se adivinan las unidades de b. Una señal que persiste al último gradiente no es necesariamente fondo.

Para actualizar el perfil k se calcula, conservando el signo,

    R_k = Y - sum(j != k) (K S(:,j)) A(j,:),
    y_k = R_k A(k,:)' / ||A(k,:)||².

TRAIn resuelve el problema con el kernel físico mediante h=eta², productos Gauss–Newton y región de confianza de Steihaug. Se normaliza el perfil candidato y se reajustan todas las amplitudes con NNLS de MATLAB. Solo se acepta si no aumenta el residuo global. No se añaden penalizaciones lambdaS/lambdaA a la matriz ni a la referencia residual.

La parada de TRAIn compara el residuo con `term_factor * ||K h_NNLS-y_k||`, con term_factor=1,05 por defecto, intervalo 1,02–1,05 y tolerancia aritmética `100*eps*||y_k||`. El residuo NNLS es una **referencia numérica operativa**, no una medida independiente de ruido. Esta regularización por parada no certifica el mínimo global ni las condiciones KKT conjuntas.

Frente a signed-v2 de DiffAtOnce, esta revisión conserva las colas pequeñas de la distribución, trata las observaciones exactamente nulas con una inicialización positiva homogénea, verifica las semillas y permite 2000 iteraciones internas por defecto. Trata explícitamente una solución NNLS nula. Por estas diferencias documentadas no se promete identidad bin a bin entre C# y MATLAB.

## Selección automática y diagnósticos

Se ajustan rangos de 1 a r_max sin usar las filas de validación. Se comparan sus errores de predicción y se escoge el menor rango elegible cuya pérdida adicional no supera un error estándar de las diferencias emparejadas frente al mejor. Después se reajusta con todas las filas. La elegibilidad exige resolver los subproblemas TRAIn y conservar factores activos. Si ninguno cumple, el resultado permanece sin resolver. Es una regla heurística con una partición, no un intervalo de confianza calibrado.

La comprobación del residuo NNLS final es independiente de esa selección. Puede quedar `train_mf_requires_review` aunque el rango predictivo sea adecuado: el NNLS independiente dispone de más libertad para ajustar ruido. No se añaden factores solo para aprobar esa comprobación. Revisar `rank_selection_resolved`, `validation_mean_loss`, `rank_trials`, `success`, `residual_target` y `subproblem_failure_details`.

En MATLAB se reserva aproximadamente un cuarto de los gradientes interiores, de forma determinista; también se acepta `validation_rows` lógico. La máscara espectral debe construirse sin esas filas. La API reserva las mismas filas que excluye al detectar señal. Sigma sirve para esa detección, pero no sustituye la referencia residual del solver revisado. La evaluación independiente necesita gradientes externos.

## Uso

```matlab
addpath('matlab');
D = logspace(log10(.02e-9),log10(5e-9),256)';
[X,D,info] = TRAIn_DOSY_MF_Signed(Y,b,D,struct('r_max',4));
% Entrada compatible con el formato de rejilla anterior, ya corregida:
[X,D_nano,info] = TRAIn_DOSY_MF(Yfull,b,[.02 5 256], ...
    struct('signal_mask',signalMask,'n_components','auto','r_max',4));
```

La máscara lógica selecciona regiones completas con señal. Las columnas excluidas son NaN, porque no se estiman. `info.A` y los diagnósticos siguen el orden de `info.process_idx`. Se rechazan opciones antiguas incompatibles de normalización, suavizado, unidades automáticas o penalizaciones. La versión V3.1 histórica sigue disponible para reproducir resultados anteriores.

Tras configurar el MATLAB con licencia según [la API](API_ES.md):

```sh
train-dosy examples/one_component/input.json resultado-revisado.json --method TRAIn-MF
```

Python y el cliente C# llaman al mismo solver MATLAB. `max_components`, de 1 a 4, limita la búsqueda; no impone el rango verdadero. HTTP 200 solo confirma que terminó el cálculo. La respuesta incluye `protocol`, `diagnostics`, S y A; `kkt` es null, porque esta implementación no calcula un certificado KKT conjunto.

## Comprobación reproducible

```matlab
addpath('matlab/tests');
report = test_train_mf_signed('verification/train_mf_signed_regression.json');
```

Las pruebas con semilla fija contienen 1, 2 y 3 perfiles anchos compartidos entre frecuencias y regiones con solapamiento. La verdad se usa solo para evaluar. Se comprueban D media, predicción externa, rango, escalado, permutación de frecuencias, colas negativas/nulas, rango insuficiente, máscaras y opciones incompatibles. Otro caso reproduce el sesgo causado por restar un mínimo que todavía contiene señal.

Son regresiones sintéticas, no validación experimental nueva. Recuperar bien la media o la atenuación no garantiza recuperar la moda o la anchura. No se consideran resueltas las limitaciones del mapa completo observadas en DiffAtOnce.
