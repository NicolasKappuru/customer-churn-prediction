# w2 · Plan: de clasificación de churn a predicción de trayectorias

Estado: propuesta, sin implementar.

## 1. Situación actual

`workshops/w2/w2_neural_models.ipynb` entrena un LSTM y un GRU *many-to-one*:

- Entrada `(batch, 90, 20)`. Solo se usa el último estado oculto `h_90`.
- `recurrent_classifier.py`: `Linear(hidden, 1)` con `OUTPUT_DIM = 1`. Devuelve un logit de `is_churn`.
- `data.py`: `y = is_churn` con forma `(batch, 1)`.
- `train.py:38`: `BCEWithLogitsLoss`. `train.py:258`: `sigmoid` al predecir.
- `tune.py:179`: Optuna minimiza el log-loss.

El enunciado pide **solo trayectorias** de comportamiento. El target pasa de ser una etiqueta (0/1) a ser los próximos H días de los canales del cliente. Eso convierte la tarea en un problema de regresión (con algunos canales binarios).

## 2. Qué son los "días futuros"

La ventana de 90 días que ya existe se parte en dos:

```text
día:     0 ........................... 82 | 83 ........ 89
         entrada (90−H = 83 días)          | target (H = 7 días)
```

| Cohorte | Entrada | Target |
|---|---|---|
| `train_feb` | 2016-11-03 a 2017-01-24 | 2017-01-25 a 2017-01-31 |
| `test` (marzo) | 2016-12-01 a 2017-02-21 | 2017-02-22 a 2017-02-28 |

Son "futuros" respecto a lo que ve el modelo, no respecto al calendario. No hace falta rehacer el export y se mantiene el split `fit/stop/tune/test`.

La alternativa de predecir el futuro real después del cutoff (marzo, en `user_logs_v2`) obliga a rehacer el export de secuencias. No se recomienda para el taller.

## 3. Horizonte

**H = 7 días.**

- Cubre una semana completa, con el ciclo semanal de escucha.
- La entrada sigue siendo larga (83 días).
- La escucha diaria es muy ruidosa: con H = 30 casi todo el error sería ruido y la entrada bajaría a 60 días.

H = 14 es aceptable si se quiere algo más ambicioso, reportando también el error en h = 7. No pasar de 14.

## 4. Por qué la salida es continua

`total_secs = 12,457` o `num_100 = 40` son magnitudes, no clases. Después de `log1p` (9.43 y 3.71) y del escalado estándar, pueden ser cualquier número real, incluso negativo. El sigmoid solo devuelve valores en (0, 1), así que los canales de cantidad necesitan salida lineal y una pérdida de distancia (MSE o Huber).

Los contadores (`num_25`, `num_100`, `num_unq`) son enteros de origen. Se tratan como continuos por su rango amplio y por las transformaciones. La alternativa (pérdida Poisson o binomial negativa sobre conteos crudos) es más correcta en teoría, pero más compleja.

## 5. Salida por canal (se predicen los 20)

`channel_specs` ya trae `encoding`, `group` y `mask`, con lo que se puede asignar el tipo de salida automáticamente.

| Canales | Qué se predice por día | Activación | Pérdida |
|---|---|---|---|
| 0–6 escucha (`num_25` … `total_secs`) | cantidad | lineal | Huber / MSE |
| 7 `active` | ¿escucha ese día? | sigmoid | BCE |
| 8 `tx_count` | ¿hay pago ese día? | sigmoid | BCE con `pos_weight` |
| 9 `tx_paid_sum`, 10 `tx_plan_days_sum` | monto y plan **si paga** | lineal | MSE solo en días con pago (máscara `tx_count`) |
| 11 `tx_cancel_count`, 12 `tx_auto_renew_count` | ¿el pago es cancelación o renovación automática? | sigmoid | BCE con máscara `tx_count` |
| 13 `tx_state_known`, 14 `tx_last_is_cancel`, 15 `tx_last_is_auto_renew`, 18 `tx_expiry_known`, 19 `tx_state_ambiguous` | flag de estado | sigmoid | BCE |
| 16 `tx_days_since_last`, 17 `tx_days_to_expire` | días | lineal | Huber |

Los montos (9–10) se predicen en dos partes: si hay pago y cuánto, condicionado al pago. Con MSE sobre todos los días el modelo aprendería a predecir casi cero, porque casi nunca hay pago.

## 6. Arquitectura y pérdida

```text
entrada (B, 83, 20) → LSTM/GRU → h_83 → BatchNorm → Linear(hidden, 7·20) → reshape (B, 7, 20)
                                                                     ├─ 9 canales lineales: valor directo
                                                                     └─ 11 canales binarios: logits → sigmoid al predecir
```

```text
L = w_esc · Huber(escucha) + w_bin · BCE(flags y eventos) + w_monto · MSE_enmascarado(montos) + w_dias · Huber(días)
```

Los pesos `w` compensan las escalas distintas de cada término. Se fijan a mano o se buscan con Optuna.

## 7. Cómo se predice el estado

**A. Directamente con su cabeza.** Simple, un solo modelo. Problema: cuando el modelo duda entre "paga" y "no paga", el MSE lo lleva al promedio de los dos escenarios y produce valores que no existen en ninguna trayectoria real.

**B. Predecir eventos y recalcular el estado** con la lógica del export (`build_lstm_sequences`): un pago predicho reinicia `tx_days_since_last` a 0 y mueve `tx_days_to_expire` unos 30 días hacia adelante. Sin pago, los contadores siguen subiendo o bajando de a uno. Es coherente por construcción, pero el recálculo es posterior al entrenamiento, porque el umbral no es diferenciable.

**Recomendación:** entrenar con A y reportar también B como versión coherente, y comparar las dos.

## 8. Ejemplo de salida (números inventados)

### Un cliente, canales de escucha

Valores en `log1p`, entre paréntesis el valor original.

| Día | Fecha | `num_100` real | `num_100` pred | `total_secs` real | `total_secs` pred |
|---|---|---|---|---|---|
| 83 | 22 feb | 3.71 (40) | 3.42 (30) | 9.43 (12,457 s) | 9.10 (8,955 s) |
| 84 | 23 feb | 3.50 (32) | 3.38 (28) | 9.20 (9,897 s) | 9.05 (8,518 s) |
| 85 | 24 feb | 0.00 (0) | 3.30 (26) | 0.00 (0 s) | 8.98 (7,942 s) |
| 86 | 25 feb | 3.95 (51) | 3.35 (27) | 9.60 (14,764 s) | 9.02 (8,267 s) |
| 87 | 26 feb | 3.40 (29) | 3.29 (26) | 9.15 (9,414 s) | 8.97 (7,863 s) |
| 88 | 27 feb | 3.64 (37) | 3.31 (26) | 9.35 (11,500 s) | 8.99 (8,022 s) |
| 89 | 28 feb | 3.81 (44) | 3.33 (27) | 9.43 (12,457 s) | 9.01 (8,184 s) |

El modelo predice una trayectoria suave cerca del nivel medio del cliente, no anticipa el día inactivo (85) y subestima los días altos.

### El mismo tipo de salida, canales de transacciones

El cliente tiene renovación automática y en el día 82 le faltan 4 días para vencer. En la realidad paga el día 86.

| Día | Pago real | p(pago) | `days_to_expire` real | pred A | pred B | `days_since_last` real | pred A | pred B |
|---|---|---|---|---|---|---|---|---|
| 83 | 0 | 0.02 | 3 | 3.1 | 3 | 27 | 27.0 | 27 |
| 84 | 0 | 0.05 | 2 | 2.0 | 2 | 28 | 28.1 | 28 |
| 85 | 0 | 0.21 | 1 | 1.4 | 1 | 29 | 27.5 | 29 |
| 86 | **1** | **0.78** | 30 | **22.5** | 30 | 0 | **6.2** | 0 |
| 87 | 0 | 0.06 | 29 | 24.0 | 29 | 1 | 7.0 | 1 |
| 88 | 0 | 0.03 | 28 | 23.1 | 28 | 2 | 8.1 | 2 |
| 89 | 0 | 0.02 | 27 | 22.0 | 27 | 3 | 9.0 | 3 |

La opción B usa el umbral 0.5 sobre p(pago). Aquí acierta todo porque el pago predicho coincide con el real. Si lo predijera un día tarde, B fallaría por mucho en esos días, mientras que A fallaría por poco en todos. Los valores de A en negrita (22.5 días, 6.2 días) son la mezcla de "renovó" con "no renovó".

## 9. Evaluación

El error por sí solo no dice si el modelo es bueno. Cada grupo se compara con predictores simples que no aprenden nada.

| Grupo | Métrica | Baseline |
|---|---|---|
| Escucha (0–6) | MAE, RMSE en `log1p`; skill = `1 − MAE_modelo / MAE_baseline` | persistencia (día 82), media de los últimos 7 días, semana anterior (días 76–82) |
| `active` | AUC, AP por día | ¿estuvo activo el día 82? |
| Pago (8) | AUC, AP por día; % de pagos acertados con ±1 día | **regla de vencimiento:** paga cuando `days_to_expire` llega a 0 con renovación automática |
| Montos (9–10) | MAE solo en días con pago | último monto pagado |
| Flags de estado | accuracy, AUC | persistencia |
| Días (16–17) | MAE en días, invirtiendo `log1p` | regla aritmética (+1 / −1 por día) |
| Coherencia (solo A) | % de días donde el estado contradice los eventos predichos | 0 % |

Además, en el grupo de escucha:

- Error por día del horizonte (h = 1…7). Debe crecer con h; si es plano, el modelo predice algo parecido a una constante.
- Error por canal.
- Error separado para días activos e inactivos, y solo sobre usuarios con logs. Muchos días valen cero (185k usuarios de febrero no tienen logs) y un modelo que predice casi cero puede tener un MSE bajo sin servir.
- Gráficas de trayectoria real contra predicha para clientes activos, clientes que se apagan y clientes inactivos.

Ejemplo de tabla resumen (números inventados):

| Modelo | MAE escucha | RMSE | Skill vs persistencia |
|---|---|---|---|
| Persistencia | 0.95 | 1.60 | 0.00 |
| Media 7 días | 0.88 | 1.32 | 0.07 |
| Semana anterior | 1.02 | 1.71 | −0.07 |
| LSTM | 0.79 | 1.18 | 0.17 |
| GRU | 0.80 | 1.19 | 0.16 |

La regla de vencimiento es un baseline fuerte para los pagos. Si el LSTM no le gana, no aprendió nada sobre pagos. Un resultado honesto puede ser que el modelo aporta en escucha y en pagos irregulares (clientes sin renovación automática), pero no en los estados.

LSTM contra GRU: reutilizar el bootstrap pareado de `src/models/evaluation/significance.py` con el error por usuario.

## 10. Relación con el w1

Los resultados no se pueden comparar directamente con el w1. El w1 clasifica churn (log-loss, ROC AUC, AP) y el w2 predice trayectorias (MAE, RMSE). En el informe hay que decir explícitamente que son tareas distintas.

Conexiones indirectas opcionales:

1. Usar features derivadas de la trayectoria predicha en el XGBoost del w1 y comparar AUC/AP con M3v2 (0.864 / 0.590). Para no mezclar información, la trayectoria tendría que predecirse a partir del cutoff (marzo real, `user_logs_v2`).
2. Comparar el error de trayectoria entre churners y no churners del test, como se hizo en el w4 con el error de reconstrucción.

## 11. Cambios en el código

| Archivo | Cambio |
|---|---|
| `src/models/recurrent_models/data.py` | entrada `x[:, 0:83, :]`, target `x[:, 83:90, 0:20]`, máscaras de días con pago |
| `src/models/recurrent_models/config.py` | `OUTPUT_DIM`, `HORIZON = 7`, pesos de la pérdida |
| `src/models/recurrent_models/recurrent_classifier.py` | cabeza `Linear(hidden, 7·20)`, `reshape`, separación por grupo según `channel_specs` |
| `src/models/recurrent_models/train.py` | pérdida compuesta con máscaras; sigmoid solo en canales binarios al predecir |
| `src/models/recurrent_models/tune.py` | Optuna minimiza la pérdida compuesta en `tune` (o el MAE de escucha); se elimina la búsqueda de umbral |
| `src/models/recurrent_models/test.py` | sin umbral; guarda las trayectorias predichas |
| `src/models/evaluation/` | módulo nuevo de regresión: baselines, métricas por grupo y por horizonte, reconstrucción de estado (opción B), gráficas |
| `w2_neural_models.ipynb` | reescribir las secciones de modelo, test y evaluación (o crear `w2_trajectory_models.ipynb`) |
| `docs/w2/group2_workshop2.tex` | reescribir las secciones de modelo y evaluación |

La evaluación es el cambio más grande. El cambio en el modelo son pocas líneas.

## 12. Decisiones pendientes

- [ ] Confirmar H = 7 (o 14).
- [ ] Confirmar que se predicen los 20 canales, o solo los 7 de escucha.
- [ ] Opción A, B o ambas para el estado.
- [ ] Reemplazar el notebook actual o crear uno nuevo.
- [ ] Qué minimiza Optuna.
