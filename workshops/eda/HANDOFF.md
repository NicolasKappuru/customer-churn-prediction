# Handoff (2026-11-14): estado del trabajo EDA → w1/w2

## Qué está hecho

- Commit `a12fa00`: EDA 00–09, `_common.py`, `DECISIONS.md`, `README.md`, `PLAN.md`, `figures/`, `workshops/PLAN_W1_W2.md` y el EDA viejo movido a `workshops/w1/eda_legacy/`.
- **Sin commit todavía:**
  - `10_per_cycle_prediction.ipynb` (ejecutada), `figures/10_k_curve.png` y la sección 10 de `DECISIONS.md`.
  - `_common.py`: los helpers aceptan un corte **por cliente**. Si `users` trae una columna `as_of`, la usan `subscription_state`, `history_block` y `member_block`; también `listening_block(users, days=daily_logs(...))`, y `filter_as_of(t, users)` filtra las transacciones. Sin `as_of` se comportan igual que antes (probado: resultado idéntico).
  - `w1/w1_tabular_models.ipynb`:
    - Se arregló el import de `DEV_COHORT` y `TEST_COHORT`, que ahora vienen de `src.models.constants`.
    - Se agregó la celda `subset7-features`, después de `validate_and_save_split`. Entrena con 7 features y escribe en `artifacts/sherlock_w1_subset7/`. Resultado: F1 en marzo de 0,42–0,47 contra 0,56–0,58 con las 108, porque no tiene ninguna feature del estado de la suscripción. `USE_SUBSET = False` vuelve al set completo.

## Resultados clave

- nb 09: las 20 features del EDA le ganan a las 108 del w1 actual con el mismo XGBoost (marzo: AUC 0,8689 vs 0,8644; AP 0,601 vs 0,593; F1 0,584 vs 0,576).
- nb 10: predicción por ciclo, con `as_of = vencimiento − k` por cliente, sobre el 89 % que vence en el mes de la etiqueta y con los mismos clientes en todos los escenarios. AP en el holdout de febrero / marzo:
  - calendario: 0,452 / 0,554;
  - k=28: 0,427 / 0,549 (peor que el calendario);
  - k=14: 0,497 / 0,594;
  - k=7: 0,559 / 0,631;
  - k=1: 0,686 / 0,691.

  El bootstrap pareado excluye el 0 en todas las diferencias. La ganancia viene de los churners con renovación automática que la apagan antes de vencer: con k=7 el recall de ese grupo pasa de 1–6 % a 22–28 %, con precisión de 0,83–0,90.
- Recomendación de la nb 10: **k = 7 por defecto** (la elección final es de negocio). Quedan abiertos dos temas: el umbral y la calibración cuando cambia la prevalencia, y qué modelo usa el 11 % fuera de alcance.

- nb 11 (auditoría de la etiqueta): `is_churn` = **no hay un pago fechado entre el vencimiento y 30 días después** (coincide en el 99,4 % de los casos). Un pago anticipado no cuenta como renovación. Los manuales que pagan por adelantado tienen 40 % de churn (26 % de los churners manuales). Es estable entre splits y **aprendible** (AUC 0,93 dentro del grupo). Decisión: mantener la etiqueta oficial y agregar al modelo por ciclo las features `paid_ahead`, `days_early`, la duración del plan del pago anticipado y los pagos del ciclo actual.

## Siguiente paso

- E3c (nb 10, sección 6): las 6 features del pago anticipado **no aportan** (ΔAP entre −0,001 y +0,002 para todo k). `days_to_expire` y el bloque del último pago ya capturan esa información, y el modelo base ya tiene AP 0,85–0,91 en los clientes que pagaron por adelantado (k=7). Se descartan. Lo de 6 % vs 42 % en el grupo con extensión de 26–30 días quedó resuelto en la nb 11, sección 5: es composición por la duración de los meses, no una mecánica de la etiqueta. Un cobro automático anticipado extiende +28 días en febrero (cae dentro del rango) y +31 en marzo (cae fuera). Con la extensión exacta, los dos meses coinciden (automático ~2 %, manual 30 días ~46 %). Consecuencia: después de un pago anticipado, `days_to_expire` mezcla k con la duración del mes. Diagnóstico A (fuera de las notebooks, script en la sesión): **no hace daño**. Los automáticos que pagaron por adelantado y tienen `days_to_expire` = 32 en marzo reciben p ≈ 0,025 y una tasa de falsos positivos de 0,05 %, porque el modelo los separa por `state` y el método de pago. Exceso de falsos positivos sobre la tasa de febrero: 0 con k=3 y 28 con k=1 (F1 +0,0002). Se descarta la codificación robusta. La meseta de marzo es de **recall/calibración**: en marzo la precisión es más alta que en febrero (0,71 vs 0,62 con k=1) y el recall más bajo (0,50 vs 0,63). En los clientes que no pagaron por adelantado, el churn de marzo es 6,4 % pero la p media es 4,75 %: el modelo subestima marzo. Ahora está en la nb 10 (secciones 6.2, 6.3 y 7). Con el umbral óptimo de marzo (oráculo) no hay meseta: F1 0,612 / 0,629 / 0,641 con k = 7 / 3 / 1. El umbral de febrero pierde 0,008 / 0,033 / 0,049. La corrección de prior por EM (Saerens 2002, sin etiquetas) recupera +0,004 / +0,009 / +0,011 y nunca empeora. Se queda corta porque estima la tasa de marzo en 6,0–6,7 % contra el 7,9 % real: el cambio no es solo de prior. Decisión: EM antes del umbral, más recalibración mensual con el último mes que tenga etiquetas maduras, o un corte por presupuesto de contactos.
- Ojo: XGBoost multihilo varía ±0,0005 de AP entre corridas idénticas. Las diferencias de ese orden son ruido.

1. **Decidido por el usuario: la prioridad es la predicción por ciclo (k)**, no el corte de calendario. E3c ya está hecho (sin ganancia). Siguiente candidato: investigar la diferencia febrero/marzo en el grupo con extensión de 26–30 días (meseta del F1 en marzo) o pasar directo al pipeline de w1 por ciclo.
2. Commit del EDA 10 y de los cambios de `_common.py`.
3. Pipeline de w1 (20 features, export v6, con `as_of` por cliente si se adopta) → constructor de ciclos de w2 (`_common.payment_cycles` existe, sin probar a fondo).

## Cómo ejecutar

- `workshops/.venv` está roto (se creó en otra ruta). Hay que usar el Python del flatpak de VS Code:
  `flatpak run --command=python3 com.visualstudio.code -` (script por stdin), o
  `flatpak run --command=sh com.visualstudio.code -c 'cd <ruta>/workshops/eda && python3 -m nbconvert --to notebook --execute --inplace NB.ipynb > cache/log 2>&1; echo exit=$? >> cache/log'`.
  No pongas `| tail` detrás: oculta el código de salida real.
- **Memoria:** la nb 10 usa ~8 GB (con `msno` como texto llegaba a 22 GB y el sistema la mataba por falta de memoria). No la corras junto a otro kernel grande, por ejemplo el w1 entrenando.
- pandas 3 + XGBoost 3.1: las categóricas van como códigos enteros con `feature_types=['c', ...]` (las categorías `str` fallan).
- La 07 lee la caché de la 06 (`cache/06_listening_features.parquet`).
