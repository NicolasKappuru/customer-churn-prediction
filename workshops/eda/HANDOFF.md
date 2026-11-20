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

## nb 12 (2026-11-14): por qué el modelo por ciclo subestima marzo

- **Causa: la etiqueta de `train_v2` (marzo) sigue otra regla que la de `train` (febrero).** No es prior, ni covariables, ni comportamiento: los clientes renuevan igual en los dos meses. Cambia cómo se etiqueta: renovar con un plan > 31 días es 2 % churn en febrero y 97 % en marzo; renovar y luego cancelar, 48 % vs 96 %.
- El desfase se concentra en manuales con plan > 31 días (2,5 % de clientes, ~50 % del déficit; churn 44–47 % → 82–83 %), renovación automática (~30 %) y manuales de 30 días (~17 %).
- Reetiquetando febrero con esas dos reglas (desde sus propias transacciones) y reentrenando igual: F1 marzo 0,592 → 0,613 (k=7) y 0,590 → 0,674 (k=1); AP 0,618 → 0,661 y 0,682 → 0,730. El umbral de febrero queda a 0,002 del óptimo de marzo.
- **EM empeora** en la población elegible completa (−0,009 / −0,005 F1). La ganancia de la nb 10 era un artefacto del filtro de "clientes comunes". Se descartan EM y la recalibración mensual.
- Corte de calendario (cohorte completa, como nb 09; sección 5.1): mismo problema (6,4 % predicho vs 9,0 % real). Reglas 1 + 2: F1 0,584 → 0,592, AP 0,601 → 0,627. Con la regla 3 exploratoria (prepagados más allá del mes siguiente = churn; 40 % en febrero, 90 % en marzo): F1 0,628. Los que vencen el mes siguiente siguen subestimados (4,3 % vs 9,0 %): esa parte de la regla no está identificada.
- **Pendiente de decisión del usuario:** la regla se encontró auditando etiquetas de marzo. Falta aprobar su uso para construir el target de entrenamiento (o confirmar la definición de `train_v2` en su fuente), y probarla en la nb 09 / pipeline de w1.
- Caché: `cache/12_feat_<cohorte>_k<k>.parquet` y `cache/12_post_<cohorte>.parquet` (la notebook los reconstruye en ~2 min si faltan; corre completa en ~10 min, ~6 GB).

## nb 13 (2026-11-14): el target reetiquetado en los dos modelos de calendario

- `13_relabelled_target_models.ipynb` entrena el M3v2 de w1 (59 features, hiperparámetros ya tuneados) y el modelo de 20 features del EDA con tres targets de febrero: oficial, reglas 1 + 2 y reglas 1 + 2 + 3. Siempre se evalúa con la etiqueta oficial de marzo y el umbral de febrero.
- F1 en marzo, oficial → 1 + 2 → 1 + 2 + 3:
  - 20 features: 0,584 → 0,592 → 0,628 (AP 0,601 → 0,627 → 0,661).
  - 59 features: 0,578 → 0,586 → 0,610 (AP 0,596 → 0,622 → 0,603).
- El modelo de 20 features gana con los tres targets.
- **Ojo con la regla 3 y `days_to_expire`:** en febrero "prepagado más allá del mes siguiente" es `days_to_expire >= 60` y en marzo es `>= 62`. El modelo de 59 features aprende el 60 y da p = 1,00 a 3.960 clientes de marzo que vencen el 29–30 de abril (churn 5,7 %): unos 3.700 falsos positivos, y por eso su AP baja con la regla 3. El de 20 usa `expiry_month` y no tiene el problema.
- Los que vencen el mes siguiente siguen subestimados en los dos modelos (4,3–4,5 % contra 9,0 %).
- Las 59 features salen del export local (108 columnas) más cuatro `trend_*` reconstruidas en la notebook. El M3v2 guardado, evaluado con esas columnas, reproduce su AP de marzo (0,596).
- Necesita `cache/12_calendar.parquet` (lo crea la nb 12). Corre en ~4 min con GPU. Resultados en `cache/13_results.csv`.
- La nota de `LABEL_FEB_VS_MAR.md` sobre una reconstrucción de `train`/`train_v2` con el script era errónea y se quitó: los parquet actuales son los oficiales, idénticos al backup `data/_backup_20261114_165908/`.

## nb 14 (2026-11-14): split solo de febrero, 70 / 15 / 15, sin marzo

- `14_february_only_split.ipynb`: mismos dos modelos y tres targets que la nb 13, pero febrero se parte en train 70 % / tune 15 % / test 15 %. Es la misma regla de hash y estratificación que el split de w1, solo cambian las proporciones. Tune sirve para early stopping, calibración isotónica (modelo de 59) y umbral. Marzo no se lee.
- Dentro de febrero los dos modelos **empatan**: F1 con la etiqueta oficial 0,562 (20) vs 0,561 (59); AP 0,601 vs 0,600. La ventaja del de 20 solo aparece al pasar a marzo.
- Contra la etiqueta oficial de febrero, las reglas 1 + 2 no cuestan nada medible (−0,001 / −0,003 F1, el IC incluye 0) y la regla 3 cuesta ~0,02 F1.
- Medidos contra su propio target, los reetiquetados dan F1 0,56 → 0,63 → 0,65. Parte de esa mejora es mecánica (tasa base más alta, y la regla 3 sale directamente de `expiry_month`), así que no sirve para comparar targets.
- Conclusión: las reglas 1 + 2 sirven como target por defecto. La regla 3 solo conviene si el modelo se evalúa con la definición de `train_v2`.
- Corre en ~3 min. Resultados en `cache/14_results.csv`.

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
