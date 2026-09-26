# Auditoría del manuscrito — manuscript-v5

Fecha: 26 de septiembre de 2026. Documento revisado: *Positive Joint Laplace Inversion: Spectral Support Selection and Identifiability in Diffusion-Ordered Spectroscopy*.

La revisión corrige afirmaciones matemáticas y descripciones del código, incorpora las declaraciones facilitadas y mejora la presentación. No modifica estimadores, datos sintéticos, estimaciones guardadas, métricas ni los ocho archivos de figuras. La versión numérica sigue siendo **v0.1.0**. La revisión del manuscrito y las nuevas comprobaciones algebraicas no constituyen un benchmark nuevo.

## Dictamen

La contribución defendible es un procedimiento auditable de selección de soporte espectral con mejora media de asignación y predicción en el banco sintético declarado. Los resultados no justifican una selección universal del número de especies, superioridad general de DOME o RAI-Net, ni validación experimental DOSY. El resumen, los resultados y las conclusiones distinguen esas afirmaciones.

El manuscrito queda más preciso y reproducible, pero sigue siendo un borrador para revisión de los autores. Permanecen por confirmar las atribuciones CRediT y los conflictos de interés; tampoco se acredita aprobación final de todos los autores ni envío a la revista.

## Hallazgos corregidos

| Prioridad | Hallazgo | Corrección y evidencia |
|---|---|---|
| Alta | En RAI de densidad se confundía la unicidad del mínimo exacto con la precisión de un resultado que termina con éxito. | La sección 4.4 distingue convexidad estricta por ridge positivo, terminación del optimizador y error numérico. Un gradiente proyectado sin escalar o un pequeño cambio de objetivo no da, por sí solo, una cota de distancia al mínimo. |
| Alta | La notación de las propuestas DOME mezclaba unidades físicas y coordenadas internas; esto volvía ambiguos los tamaños de división y los umbrales numéricos. | La sección 4.7 introduce t=b D_ref, delta=d/D_ref y la coordenada logarítmica normalizada. Curvatura, contraste espectral y conservación de momentos se expresan en esas coordenadas; la conversión a d físico se efectúa al final. |
| Alta | La interpretación global de mejora de selección podía ocultar regresiones en ciertos criterios. | El resumen indica que el acierto de orden de DOME baja de 26/36 a 25/36 y que la masa no emparejada aumenta. La mejora común se limita a asignación espectral y predicción media. |
| Media | La regularización de curvatura de TRAIn-MF no bastaba para justificar unicidad de S. | La nueva Proposición A1 da una condición suficiente y un contraejemplo explícito de degeneración con dos filas de A iguales. No se atribuye esta degeneración al ejemplo guardado. |
| Media | Las identidades de selección suponían ruido iid; no podían trasladarse directamente a errores correlacionados. | La nueva Proposición A2 deriva las formas con covarianza especificada. Se aclara que no se implementaron ni validaron en el benchmark congelado. |
| Media | Faltaban detalles operativos de RAI de densidad y Haar necesarios para reproducir la formulación. | Se explicitan las escalas por columna de entrenamiento, su conservación en el reajuste, la reconstrucción del grafo con todas las filas de reconstrucción y el MSE de validación normalizado por ruido. |
| Media | Se llamaba «ablación» a una comparación neuronal que cambiaba objetivo de entrenamiento, anchura y banco de entrenamiento. | Se denomina comparación de políticas. El panel nuevo consta de 18 casos atómicos con señal, tres nulos y tres anchos; no es el panel primario de 36 casos. Un intervalo que incluye cero no demuestra equivalencia ni ausencia de efecto. |
| Media | La descripción del registro del Algoritmo 1 sugería que se exportaban todos los diagnósticos de elegibilidad. | El pseudocódigo enumera los campos realmente registrados. Distingue la comprobación interna de estacionariedad de las variables guardadas en el registro. El orden elegido y el límite configurado tampoco prueban ausencia de tasas adicionales. |
| Presentación | La figura DOSY discreta original contenía una etiqueta vertical de color cortada. | El manuscrito recorta únicamente ese margen y repone el significado completo como etiqueta horizontal. Los archivos originales, valores, mapas, umbral visual y escala común permanecen intactos. |

Las descripciones revisadas se contrastaron con los archivos identificados en [method_formulations.json](evidence/method_formulations.json). La transcripción de variantes históricas no implica que todas estén implementadas en la API pública.

## Desarrollo matemático añadido

Para A fijo y lambda_S>0, la segunda variación de la función TRAIn-MF en una dirección factible U es

    ||K U A||_F² / p + lambda_S ||L U||_F².

Si la intersección de KUA=0, LU=0 y 1^T U=0 contiene solo U=0, el subproblema de S es estrictamente convexo sobre el espacio afín del simplex y tiene mínimo único. Es una condición suficiente; no es una caracterización necesaria de unicidad en un óptimo de frontera. Tampoco demuestra unicidad de los factores conjuntos o identificación química.

Si A tiene dos filas iguales y v pertenece al núcleo de L con suma cero, U=[v,-v] conserva predicción y penalización. Para perfiles estrictamente positivos, una perturbación suficientemente pequeña mantiene la factibilidad. La normalización y la suavidad pueden coexistir con pares de perfiles indistinguibles.

Para ruido vectorizado por columnas con covarianza Omega, un contraste fijo w tiene varianza w^T Omega w. Si Omega=Gamma ⊗ Sigma_b y w=g ⊗ v, esta varianza es (g^T Gamma g)(v^T Sigma_b v). El crecimiento habitual con la raíz del número de frecuencias necesita la correspondiente hipótesis de independencia.

Para dos predicciones fijas P,Q, peso simétrico fijo W y delta=vec(P-Q), la diferencia de pérdidas cuadráticas tiene varianza 4 delta^T W Omega W delta. Los términos cuadráticos del ruido se cancelan. Estas identidades requieren las hipótesis declaradas; reutilizar datos para elegir tasas, grupos, covarianza y candidato no produce automáticamente una prueba calibrada.

Las demostraciones figuran en el [Apéndice E](sections/audit_extensions.tex). Se añadieron siete comprobaciones deterministas a las veinte existentes, incluyendo el contraejemplo de curvatura y las identidades de covarianza. Son comprobaciones de álgebra finita, no sustitutos de una demostración, estudio de convergencia o experimento de recuperación.

## Comprobación de resultados congelados

Se recalcularon las agregaciones desde el CSV por problema, sin ejecutar los estimadores. La comparación con las tablas y los seis intervalos bootstrap guardados coincide con tolerancia numérica. Se usa la misma convención: remuestreo de 36 problemas pareados completos, 10 000 réplicas y semilla 9925.

| Método | Orden correcto | Orden y todas las tasas | Fuga espectral | Masa no emparejada | RMSE limpio / sigma |
|---|---:|---:|---:|---:|---:|
| RAI | 17/36 | 17/36 | 5.166% | 6.774% | 0.2870 |
| RAI-S | 25/36 | 23/36 | 1.728% | 7.257% | 0.2330 |
| DOME | 26/36 | 22/36 | 3.857% | 7.019% | 0.2591 |
| DOME-S | 25/36 | 23/36 | 1.728% | 7.257% | 0.2330 |

Todos los resultados atómicos guardados de los paneles nuevo y de transferencia tienen indicador de éxito. Cada política RAI-Net tiene un fallo numérico en el mismo caso archivado, `r1_snr50_rep2`; ambos resultados siguen incluidos en sus respectivas medias. Los indicadores de éxito usan reglas específicas de cada método y no son certificados intercambiables de precisión.

No se han repetido en esta revisión los 144 ajustes de soporte, la comparación MATLAB de 100 columnas ni los cinco tests numéricos históricos. Sus cifras se citan como evidencia archivada. Las comprobaciones ejecutadas ahora son las agregaciones, álgebra, identidades de archivos y validación documental descritas aquí.

## Autoría y declaraciones

1. Victor Valdivieso: afiliación 1; mcv598@inlumine.ual.es. No se le inventa ORCID.
2. Ignacio Fernández: afiliación 2; ORCID 0000-0001-8355-580X; ifernan@ual.es.
3. Francisco Manuel Arrabal-Campos: afiliaciones 1 y 2, autor de correspondencia; ORCID 0000-0002-5510-6297; fmarrabal@ual.es.

Se conservan las denominaciones institucionales facilitadas y se insertan los cinco identificadores de financiación exactamente. Los ORCID tienen checksum válido y enlaces activos en el PDF; la verificación del checksum no autentica por sí sola una identidad.

CRediT se redacta como **propuesta basada en el orden de firma**, por petición del usuario, pendiente de ratificación de los tres autores. No se atribuye adquisición de fondos sin confirmación. No se inventa una declaración de ausencia de conflictos.

Se eliminan los agradecimientos. En la actualización manuscript-v6, la declaración en metodología identifica Codex como ayuda para programar el repositorio TRAIn-DOSY y Trinka como asistencia para mejorar la claridad y el estilo del manuscrito, sin mencionar modelos ni versiones. Este cambio de redacción fue solicitado por el autor de correspondencia; la auditoría matemática de v5 se conserva. La [política editorial de MDPI](https://www.mdpi.com/ethics) pide también información sobre herramientas generativas en Acknowledgments cuando el uso supera la corrección lingüística. Por tanto, la ubicación final de esa declaración es un punto editorial pendiente para el envío; la retirada solicitada no se presenta como conformidad completa con esa política. No se ha contactado con la revista.

## Reproducibilidad y límites pendientes

- Se verifican 22 fuentes preservadas, las 27 identidades fuente/formulación disponibles en el entorno del autor, 82 archivos numéricos sin cambios desde v0.1.0 y los ocho archivos de figuras sin cambios desde manuscript-v4.
- Se conservan 48 referencias, todas citadas. Las 46 referencias con DOI coinciden con los metadatos registrados en la caché del **25 de septiembre de 2026**; esta auditoría no simula una nueva consulta de Crossref.
- El PDF tiene 38 páginas. Se inspeccionaron todas en hojas de contacto y, a tamaño de página, portada, Algoritmo 1, figuras DOSY, declaraciones y nuevas proposiciones. La compilación no presenta desbordamientos, referencias indefinidas ni destinos duplicados.
- La máscara histórica comparte observaciones con la validación interna. No se puede afirmar independencia condicional exacta después de esa selección; las ocho adquisiciones externas se mantienen fuera del ajuste y selección.
- El límite de cuatro tasas restringe la búsqueda. El rango resuelto por ruido no es una cota superior universal del número de especies.
- La evidencia actual no incluye validación experimental nueva, covarianza estimada, un diseño factorial que aísle cada mejora, ni un benchmark externo completamente armonizado. Los casos débiles y proporcionales siguen siendo limitaciones visibles.
- RAI-S y DOME-S prácticamente coinciden en este banco. Esto no constituye dos réplicas independientes de una misma mejora.

Comandos de comprobación local:

```sh
python paper/scripts/check_formulations.py
python paper/scripts/audit_manuscript.py
```

Para contrastar también las fuentes históricas que no están distribuidas en el repositorio público:

```powershell
python paper/scripts/audit_manuscript.py --local-archive 'E:/ARTICULOS-CIENTIFICOS/20240905_OLIGOQUITOSANO/CODIGO_MEJORADO'
```

Resultados legibles por máquina: [formulation_checks.json](../verification/formulation_checks.json), [manuscript_audit_v5.json](../verification/manuscript_audit_v5.json) y su comprobación documental actualizada [manuscript_audit_v6.json](../verification/manuscript_audit_v6.json) y [paper_audit.json](../verification/paper_audit.json). Los checks que necesitan objetos Git o el archivo privado se declaran omitidos cuando esos recursos no están disponibles; no se cuentan como superados. La revisión visual es una comprobación separada de los scripts.
