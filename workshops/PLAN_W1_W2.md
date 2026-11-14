# Plan W1 y W2 a partir del EDA (`workshops/eda/`)

Este plan traduce `eda/DECISIONS.md` a cambios concretos en w1 (tabular) y w2 (secuencial).
Todas las cifras salen de las notebooks 00–08 y de un sondeo sobre los ciclos de pago, calculado
sobre `fit` (744.699 clientes, corte 2017-01-31).

## 0. Lo que el EDA dejó fijo

| Hallazgo | Consecuencia |
|---|---|
| El estado de la suscripción al corte explica casi todo (AP 0,51 solo; sin él el AP cae 0,20) | Es el núcleo de ambos modelos |
| El historial de pagos (antigüedad como pagador, pagos, lapsos) suma +0,036 AUC, y **no cabe en 90 días** | w2 tiene que ver todo el historial, no una ventana |
| La escucha importa sobre todo en el pago manual, y por **recencia** (silencio reciente), no por forma | En w2 basta poca resolución diaria; en w1, 5 features |
| La dinámica temporal está **anclada al ciclo de cobro** (pago → vencimiento), no al calendario (nb 07) | El tensor de w2 se organiza por ciclos |
| El 90 % de los ciclos cerrados dura 28–31 días; mediana de 15 ciclos por cliente; 26 ciclos cubren al 98,9 % | Secuencia de ciclos con K = 26 |
| Ciclo actual al corte: mediana de 14 días transcurridos, p95 = 30 | Detalle diario del último mes (31 días) |
| Los logs de escucha y las transacciones empiezan el 2015-01-01 | Se puede agregar la escucha por ciclo sobre todo el historial; el 31 % tiene historial truncado (flag) |
| 20 features tabulares le ganan a las 108 actuales (AUC 0,900 vs 0,898; AP 0,597 vs 0,588) | w1 se reescribe con esas 20; además son el baseline de w2 |

---

## 1. W1 (tabular)

### 1.1 Pipeline de features (`w1/src/data_processing`)

Producir exactamente la tabla de 20 features de la sección 7 de `eda/08_integration_and_drift.ipynb`. La
implementación de referencia ya existe y está probada en `eda/_common.py` (`clean_transactions`,
`subscription_state`, `state_block`, `history_block`, `listening_block`, `member_block`).

| Módulo | Cambio |
|---|---|
| `members.py` | Quitar la imputación por media y la regla de canales negativos. Producir `via`, `profile_state`, `age_band`, `bd` (con NaN) y `tenure_days` (NaN si falta o es posterior al corte) |
| `transactions.py` | Limpieza de la nb 02: anular los ceros de 2015 y las fechas de vencimiento inválidas, colapsar los registros por día, y tratar las cancelaciones como cambios de estado. Salidas: `state`, `expiry_month` (por mes calendario), `plan_band`, `price_level`, `price_type`, `pm_group`, `payer_days` (con tope de 730), `n_payments`, `n_lapses`, `total_lapse_days`. Se eliminan las ~19 agregaciones actuales |
| `user_logs/monthly.py` | Reemplazar por un `listening.py` que produzca las 5 features (`has_logs`, `days_since_last_log`, `longest_gap`, `active_days_30`, `secs_30`). Días de más de 24 h reescalados; **sin** recorte p99.9 |
| `params.py` | El único estado que se aprende en `fit` son los métodos de pago con ≥ 1.000 clientes. Se borran los parámetros `age_mean`, `registered_via_mode` y `log_clip_p999` |
| `pipeline.py` | Ensamblar las 20 columnas y exportar `export-tabular-v6` |
| Tests | Que no entre ningún registro posterior al corte (nb 00), que `state` no dependa del orden del archivo (nb 02), y un PSI entre febrero y marzo menor a 0,1 en cada feature (nb 08) |

### 1.2 Modelos (se mantienen M1–M3)

| Modelo | Entrada |
|---|---|
| **M1 logística** | Variables numéricas en bandas (las de la nb 08) + one-hot + 7 interacciones: `state × expiry_month`, `via × profile_state`, e historial y recencia × "pago manual". Validación cruzada en el EDA: AUC 0,883 / AP 0,568 |
| **M2 random forest** | Las 20 features; categóricas en one-hot u ordinales |
| **M3 XGBoost** | Las 20 features; categóricas nativas (`enable_categorical`) o one-hot. Validación cruzada en el EDA con gradient boosting: AUC 0,900 / AP 0,597 |

Optuna, splits (`fit` / `stop` / `tune`), umbral sobre `tune` y evaluación única sobre marzo: **igual que hoy**.
Ojo con las diferencias de tasa base (6,4 % en febrero contra 9,0 % en marzo): reportar AP y log-loss,
no solo F1 con un umbral fijo.

### 1.3 Informe W1

Reescribir la sección de EDA (hoy describe la serie vieja y dice que el EDA se hizo sobre marzo) y la de
feature engineering, con la secuencia población → limpieza → estado → historial → escucha → miembros →
selección. Agregar la tabla "w1 actual vs propuesto".

---

## 2. W2 (secuencial): tensor organizado por ciclos de pago

### 2.1 Por qué cambiar la ventana de 90 días

1. La señal temporal vive en el **ciclo de cobro**. En calendario se ve borrosa, porque cada cliente vence un día distinto (nb 07, curvas alineadas al vencimiento).
2. Lo que le falta al LSTM actual es el **historial**. Con solo 90 días + estado, el modelo llega a AUC 0,875; con historial, a 0,900 (nb 07).
3. Una ventana de 90 días diarios es a la vez demasiado larga (más allá de ~45 días agrega ruido) y demasiado corta (no ve los 25 meses de historial).

La alternativa es **una secuencia de ciclos** que cubra todo el historial, más un **detalle diario del último mes**, más **features estáticas**.

### 2.2 Definición de ciclo

- **Inicio:** un *día de pago*, es decir, un día con al menos un registro `is_cancel = 0`, después de colapsar los registros por día (regla de la nb 02).
- **Fin:** el día antes del siguiente día de pago, o el corte si es el ciclo actual (que queda incompleto).
- **Vencimiento del ciclo:** el último vencimiento válido del día de inicio. Si dentro del ciclo hay una cancelación, se registra el vencimiento posterior a la cancelación.
- **Lapso previo:** `inicio − vencimiento del ciclo anterior`. Si es mayor que 0, el cliente estuvo sin suscripción esos días.
- **Escucha del ciclo:** los días de log en `[inicio, fin]`, asignados al ciclo con un `join_asof` por cliente y fecha.
- Se toman los **últimos K = 26 ciclos** (cubre al 98,9 %), con padding a la izquierda y máscara. Los clientes sin pagos (0,2 %) tienen longitud 0 y un flag estático.

### 2.3 Rama A: secuencia de ciclos `[N, 26, ~14]`

| Canal (por ciclo) | Codificación | Respaldo |
|---|---|---|
| `cycle_days` (en el actual, días transcurridos) | log1p | nb 02 (el vencimiento se acumula) |
| `is_current` | flag | — |
| `plan_days` | log1p, 0 + flag si falta (ceros de 2015) | nb 02, 03 |
| `paid_per_30d` | paid / plan × 30, escalado | nb 03 (el precio importa en auto-renew) |
| `is_free`, `is_discounted`, `price_unknown` | flags | nb 02, 03 |
| `auto_renew` | flag | nb 03 |
| `cancel_in_cycle` | flag | nb 02, 03 |
| `lapse_days_before` | log1p | nb 04 (lapsos) |
| `listen_active_share` | días activos / días del ciclo | nb 06 |
| `listen_log_secs_per_day` | media de log1p(segundos), con 0 en días sin escucha | nb 05, 06 |
| `listen_active_last7` | días activos en los últimos 7 días del ciclo | nb 07 (caída antes del vencimiento) |
| `listen_longest_gap_share` | racha de silencio más larga / días del ciclo | nb 06 |
| `days_to_expire` (solo el ciclo actual; 0 en los demás) | signed log1p | nb 03, 07 |

Lo que el GBM tabular necesita como features explícitas (`n_payments`, `n_lapses`, `payer_days`) aquí sale
naturalmente de la propia secuencia: el número de tokens y los lapsos entre ellos.

### 2.4 Rama B: detalle diario del ciclo actual `[N, 31, 8]`

Los últimos 31 días hasta el corte, que cubren el ciclo actual del 95 % de los clientes.

| Canal | Codificación |
|---|---|
| `active` | flag |
| `log_secs`, `log_num_25`, `log_num_100` | log1p (días de más de 24 h reescalados) |
| `day_in_cycle` | días desde el último pago / 31 |
| `days_to_expire` | signed log1p |
| `payment_today`, `cancel_today` | flags |

Captura la recencia y la caída dentro del ciclo que la secuencia de ciclos resume con poca resolución.

### 2.5 Rama C: estáticas `[N, ~15]`

`via`, `profile_state`, `age_band` (embeddings o one-hot), `bd` (NaN → 0 + flag), `tenure_days`,
`history_left_censored` (primer registro en enero de 2015) y `has_member_record`. Opcionalmente, el
`state` y el `expiry_month` actuales como refuerzo, aunque ya van en el último token.

### 2.6 Arquitectura

```text
ciclos [N,26,14] --(máscara)--> GRU/LSTM --> h_ciclos
diario [N,31,8]  ------------> GRU/LSTM --> h_diario
estáticas [N,15] ------------> MLP     --> h_estatico
concat(h_ciclos, h_diario, h_estatico) --> MLP --> logit
```

Pérdida BCE (con `pos_weight` o sin él; se evalúa con AP y log-loss). Optuna sobre el tamaño oculto, las
capas, el dropout, la tasa de aprendizaje y el weight decay; early stopping sobre `stop` y umbral sobre
`tune`, igual que hoy.

### 2.7 Experimentos (todos con el mismo split; marzo se evalúa una sola vez al final)

| Id | Entrada | Pregunta |
|---|---|---|
| E0 | LSTM actual (90 días diarios, sin estáticas) | Referencia de w2 actual |
| E1 | Solo ciclos (A) | ¿La secuencia de ciclos recupera el historial? |
| E2 | A + C | ¿Cuánto suman las estáticas? |
| E3 | A + B + C (**propuesto**) | ¿El detalle diario agrega algo sobre los ciclos? |
| E4 | E3 con GRU contra LSTM | Elegir la celda |
| B1 | GBM con las 20 features de w1 (mismo split) | **Vara de comparación** |

**Expectativa honesta (nb 07):** la forma diaria no le agregó nada a un GBM con buenos resúmenes. El
criterio de éxito de w2 es **igualar a B1** (dentro del intervalo de bootstrap) aprendiendo el historial
desde la secuencia, no "ganarle por mucho". Si E3 ≈ B1, la conclusión del informe es que el
modelo secuencial llega por sí solo a lo que en w1 requirió feature engineering; eso también es un resultado.

### 2.8 Construcción (`w2/src/data_processing/sequences`)

1. Reusar la limpieza de transacciones de w1 (un solo módulo compartido entre w1 y w2).
2. Construir la tabla de días de pago → tabla de ciclos por cliente (con los últimos 26 ciclos).
3. Logs: filtrar a los usuarios de la cohorte, a ≤ corte y a fechas ≥ 2015-01-01. Hacer un `join_asof` (`by = msno`, sobre la fecha) contra los inicios de ciclo y agregar por `(msno, ciclo)`. **Procesar por bloques de meses o de usuarios**: `user_logs` tiene 392 millones de filas.
4. Rama diaria: los últimos 31 días (reusar `clean_user_days` con la ventana corta).
5. Exportar `export-sequential-v6`: `cycles.npy [N,26,F]`, `cycle_lengths.npy`, `daily.npy [N,31,8]`, `static.npy`, `labels`, y la metadata de canales.
6. Escalado: `log1p` / `signed log1p` + estandarización con media y desviación de `fit` por canal. **Sin** recorte por cuantiles.
7. Tests de causalidad: ningún dato posterior al corte (adaptar `causal_checks.py`).

### 2.9 Riesgos

| Riesgo | Mitigación |
|---|---|
| El `join_asof` sobre 392 millones de filas no cabe en memoria | Procesar por bloques de usuarios (por ejemplo, 100k) en streaming |
| Planes largos (1 token = 410 días) | `cycle_days` y `plan_days` como canales; recortar `cycle_days` en 400 |
| Historial truncado en 2015-01 (31 %) | Flag estático `history_left_censored` |
| La cohorte de marzo tiene otro calendario (meses de 31 días) | Todo relativo al ciclo y a `expiry_month`, no a días fijos (nb 08) |

---

## 3. Orden de trabajo sugerido

1. Commit del EDA (`workshops/eda/`, `eda_legacy`, `.gitignore`).
2. **W1:** pipeline de 20 features → export v6 → M1–M3 → comparación con el w1 actual → informe.
3. **W2:** limpieza compartida → tabla de ciclos → tensores v6 → E1–E4 contra B1 → informe.
