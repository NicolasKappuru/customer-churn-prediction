# Plan del EDA (KKBox churn) — v0, para revisar antes de escribir notebooks

Este EDA reemplaza a `workshops/w1/eda/`. Allí las notebooks 01–06 venían del kernel de Rizvi
y se corrieron sobre marzo, que es el test. Las 07–10 describían features que ya estaban construidas.
La regla de esta versión es que **los datos crudos deciden las features, y no al revés**.

## Reglas comunes (aplican a todas las notebooks)

| Regla | Detalle |
|---|---|
| Población | Cohorte de febrero (`train.parquet`, 992.931 clientes). **Solo el split `fit`** (~744.699), reproducido con el mismo hash del pipeline (`salt110:`, estratificado por `is_churn`). `stop`, `tune` y marzo no se miran con etiqueta. |
| Corte temporal (as-of) | Solo se usan registros con fecha `<= 2017-01-31`. Esto incluye `transactions_v2`, que trae transacciones de febrero y marzo: sin el filtro se cuela la etiqueta. |
| Unidad de análisis | El cliente (`msno`). Las vistas por registro solo se usan para calidad de datos. |
| Muestras | Si una notebook necesita muestrear, es **aleatorio por cliente** con semilla fija. Nunca `n_rows=1_000_000`. |
| Gráfico estándar vs churn | Tasa de churn por nivel o decil, con IC 95 % de Wilson, n por barra y línea punteada en la tasa base (≈6,39 %). |
| Cierre de cada notebook | Una tabla **Hallazgo → Decisión**. Cada decisión es `feature nueva`, `transformación`, `limpieza`, `descartar` o `sin acción`. Esas filas se copian a `DECISIONS.md`. |
| Marzo | Solo aparece en la 08, **sin etiqueta**, para comparar distribuciones entre febrero y marzo (drift). |

Archivos de soporte:

- `_common.py`: rutas, `load_fit_population()`, filtros as-of, `churn_rate_by()`, `plot_churn_rate()`, `save_fig()`.
- `figures/`: salida de las figuras, con el prefijo de la notebook (`01_age_bands.png`, …).
- `DECISIONS.md`: tabla única de decisiones. Es el contrato con `data_processing` de w1 y w2.

---

## 00 · Población, etiqueta y cobertura

**Pregunta.** ¿A quién estamos prediciendo, qué significa `is_churn` y con qué información contamos de cada cliente?

| Variable | Fuente |
|---|---|
| `msno`, `is_churn` | `train` |
| `split` (fit/stop/tune) | Hash del pipeline |
| `has_member_record` | `members_v3` |
| `has_transactions` (≤ cutoff) | `transactions` + `transactions_v2` |
| `has_logs_90d` (2016-11-03 … 2017-01-31) | `user_logs` + `user_logs_v2` |
| `last_expire_date` (≤ cutoff) | Transacciones |

Análisis:
1. Tamaño y tasa de churn de cada split, para verificar que la estratificación quedó bien.
2. Matriz de cobertura: las 8 combinaciones de `has_member × has_txn × has_logs`, con su n y su tasa de churn.
3. **Chequeo de la etiqueta.** Distribución de `last_expire_date` de la cohorte. ¿Vence en febrero como dice la definición de KKBox? ¿Qué porcentaje vence antes o después?
4. `days_to_expire` al corte vs churn. Es solo una primera vista; el análisis completo va en la 03.

Decisiones esperadas: flags de cobertura, cómo tratar a los clientes sin miembro o sin logs, y si la etiqueta está alineada con el vencimiento.

## 01 · Members (demografía)

**Pregunta.** ¿Qué señal tiene el perfil del cliente y cómo hay que codificarlo?

| Variable cruda | Qué se analiza |
|---|---|
| `bd` | Válida (1–100), `0`, inválida (<0 o >100), ausente. Churn por banda (1–17, 18–22, 23–29, 30–39, 40+). Relación monotónica o con umbral. |
| `gender` | Nulo / male / female vs churn. Co-ausencia con `bd=0`. |
| `registered_via` | Churn por nivel, niveles raros (n < ~1.000) y valores negativos. |
| `city` | Churn por nivel y niveles raros. ¿`city=1` coincide con el perfil vacío? |
| `registration_init_time` | `tenure_days` al corte (deciles y bandas: <30 d, <6 m, <1 a, 1–3 a, 3+ a), año de registro y mes de registro. |

Análisis específicos:
1. **Patrón de ausencia.** Tabla cruzada de `bd=0` × `gender` nulo × `city=1`. ¿Es un solo fenómeno, "perfil incompleto"?
2. **Edad.** Comparar tres codificaciones por AUC univariado y por tasa por bin: (a) media + flag, que es lo actual, (b) bandas categóricas con "desconocido", (c) `bd` con un centinela.
3. Interacción `registered_via × perfil incompleto`: ¿el canal explica el vacío?
4. Clientes nuevos (registro en 2017 o tenure < 30 d).

Decisiones esperadas: codificación de la edad, flag `profile_incomplete`, si `gender` se usa solo como flag de ausencia, agrupación de niveles raros, y forma de `tenure`.

## 02 · Transacciones — calidad por registro

**Pregunta.** ¿Qué registros están sucios o son ambiguos, y qué reglas de limpieza necesitamos?

Variables: `payment_method_id`, `payment_plan_days`, `plan_list_price`, `actual_amount_paid`,
`is_auto_renew`, `transaction_date`, `membership_expire_date`, `is_cancel`. Todas filtradas a ≤ cutoff y a los usuarios del fit.

Análisis:
1. Duplicados exactos, y duplicados "casi exactos" (mismo día, distinto `_ord`).
2. `payment_plan_days = 0`, `plan_list_price = 0`, `actual_amount_paid = 0`, pagado > lista.
3. `membership_expire_date < transaction_date`, y vencimientos absurdos (años 2018+).
4. Semántica de `is_cancel`: ¿qué pasa con el vencimiento en la transacción siguiente?
5. Varias transacciones el mismo día: ¿cuál es el estado "final"?
6. Catálogo de planes: combinaciones frecuentes de (`plan_days`, `list_price`), es decir, el precio por día.

Decisiones esperadas: reglas de deduplicación, de valores inválidos y de desempate del mismo día. Estas reglas son compartidas por w1 (tabular) y w2 (secuencias).

## 03 · Transacciones — estado de la suscripción al corte

**Pregunta.** ¿Cuánto explica el estado *actual* de la suscripción?

| Variable (última transacción ≤ cutoff) | Tipo |
|---|---|
| `last_is_auto_renew`, `last_is_cancel` y su combinación de 4 niveles | categórica |
| `last_plan_days` | discreta (bins 0/7/30/90/180/360+) |
| `last_list_price`, `last_paid`, `last_discount`, `last_price_per_day` | continua |
| `last_payment_method_id` | categórica (con niveles raros) |
| `days_to_expire`, `days_since_last_transaction` | continua (en bandas) |
| `is_free_or_trial` (paid = 0) | binaria |

Análisis: churn por nivel o decil, interacción `auto_renew × cancel`, `payment_method × auto_renew`
(¿el método solo es un proxy de la auto-renovación?) y AUC univariado.

Decisiones esperadas: qué variables de estado entran y con qué codificación, y si `payment_method` aporta más allá de `auto_renew`.

## 04 · Transacciones — historial

**Pregunta.** ¿El historial agrega algo después de conocer el estado actual?

Candidatas (todas ≤ cutoff, y para cada una se compara "todo el historial" vs "últimos 6 o 12 meses"):

- Volumen: `n_transactions`, `total_paid`, `total_plan_days`.
- Inestabilidad: `n_payment_methods`, `n_plans`, `n_prices`, cambios de método o plan.
- Cancelaciones: `n_cancels`, `ever_cancelled`, `days_since_last_cancel`.
- Lapsos: número de "huecos" (vencimiento → siguiente pago > 0 días) y hueco máximo.
- Antigüedad pagadora: días desde la primera transacción.
- Descuentos: `share_discounted`, `mean_discount`.

Análisis:
1. Churn por decil de cada candidata.
2. **Valor incremental.** Churn por decil *dentro* de cada nivel de `last_is_auto_renew`, para ver si la señal sobrevive al estado.
3. Matriz de Spearman entre candidatas. De cada bloque con |ρ| ≥ 0,9 queda una sola.

Decisiones esperadas: lista corta de agregados del historial (que debería reemplazar las ~29 actuales) y la ventana temporal.

## 05 · Logs diarios — calidad y escala

**Pregunta.** ¿Cómo se ven los contadores diarios y qué limpieza o transformación necesitan?

Variables: `date`, `num_25`, `num_50`, `num_75`, `num_985`, `num_100`, `num_unq`, `total_secs`, en la ventana
de 90 días. Se usa una **muestra aleatoria de ~100k clientes del fit, con todos sus días**.

Análisis:
1. Filas (`msno`, `date`) duplicadas, negativos, `total_secs` no finito o enorme.
2. Consistencia: `num_unq` ≤ Σ num_*; `total_secs` vs canciones (segundos por canción).
3. Cuantiles por canal (p50, p99, p99.9, máx) y comparación de límites de recorte.
4. Forma de las distribuciones crudas vs `log1p`.
5. Días activos por cliente en la ventana: ¿es bimodal? ¿Hay clientes con 0 días?

Decisiones esperadas: reglas de limpieza, límites de recorte y la transformación de los canales. Esto sirve tanto para w1 como para la entrada del LSTM.

## 06 · Logs — agregados por cliente

**Pregunta.** ¿Qué resumen del comportamiento de escucha separa churners de no churners?

Candidatas, calculadas para las ventanas de 7, 14, 30 y 90 días antes del corte:

- Actividad: `active_days`, `days_since_last_log`, `longest_inactive_streak`.
- Volumen: `total_secs`, `total_songs`, `num_unq` (por día activo).
- Hábito: `skip_ratio` (num_25 / total), `completion_ratio` (num_100 / total), `unq_ratio`.
- Tendencia: últimos 30 d vs 30 d previos (ratio), y pendiente semanal.

Análisis: churn por decil, valor incremental dentro de `last_is_auto_renew` (igual que en la 04), elección de
ventana y redundancia (Spearman).

Decisiones esperadas: qué ventanas y agregados de escucha entran a w1, que deberían ser menos y más justificados que los `trend_*` actuales.

## 07 · Trayectorias temporales (motiva w2)

**Pregunta.** ¿Hay una dinámica *temporal* antes del churn que justifique un modelo secuencial de 90 días?

Análisis:
1. Curva diaria media (con IC) de `active`, `total_secs` y `skip_ratio` para churners vs no churners, en t = −89…0 respecto al corte.
2. La misma curva **alineada al vencimiento** (días antes de `last_expire_date`).
3. Tasa de churn según cuándo fue el último día activo (t = −89…0).
4. Eventos de transacción en la ventana: ¿cuándo ocurren los pagos y cancelaciones de los churners?
5. ¿En qué momento se separan las curvas? Eso define el largo útil de la secuencia (30, 60 o 90 días).

Decisiones esperadas: largo de la secuencia, canales que valen la pena, y si el LSTM necesita una rama estática (edad, canal, estado de la suscripción).

## 08 · Integración, redundancia y drift

**Pregunta.** ¿Cuál es el set final de features y es estable entre febrero y marzo?

1. Se juntan las candidatas aprobadas en las notebooks 01–06.
2. Spearman global y eliminación de redundancias.
3. Información mutua y AUC univariado sobre el fit, como ranking de referencia (no de selección automática).
4. Interacciones de negocio: edad × auto_renew, perfil incompleto × método de pago.
5. **Drift sin etiqueta.** PSI de cada feature entre febrero y marzo.

Salida: la lista final en `DECISIONS.md` (nombre, definición, origen, codificación, notebook que la justifica). Eso es lo que se implementa en `w1/src/data_processing` y `w2/src/data_processing`.

---

## Orden de trabajo sugerido

`00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`. La 01 es independiente de las demás. La 05 y la 06 son las más pesadas (6,8 GB de logs), así que conviene precalcular el user-day del fit una sola vez en `_common.py` (parquet en caché).
