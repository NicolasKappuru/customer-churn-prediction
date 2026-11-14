# Handoff (2026-11-14): estado del trabajo EDA → w1/w2

## Qué está hecho (sin commit todavía)

- `workshops/w1/eda/` se movió a `workshops/w1/eda_legacy/` (con `git mv`) y su README quedó marcado como legacy.
- `workshops/eda/`: notebooks 00–09 ejecutadas sin errores, más `_common.py`, `DECISIONS.md` (secciones 00–08), `README.md`, `PLAN.md` y `figures/`.
- `workshops/PLAN_W1_W2.md`: plan de w1 (20 features) y w2 (tensor por ciclos de pago: ciclos `[N,26,~14]` + diario `[N,31,8]` + estáticas).
- `.gitignore`: se agregó `workshops/eda/cache/`.

## Resultados clave

- 20 features del EDA contra las 108 del w1 actual, con el mismo XGBoost (nb 09):
  - Febrero holdout: AUC 0,9009 vs 0,8992; AP 0,601 vs 0,596.
  - Marzo: AUC 0,8689 vs 0,8644; AP 0,601 vs 0,593; F1 0,584 vs 0,576.
  - El bootstrap pareado excluye el 0 en todas las diferencias.
- Techo del F1 (~0,56–0,58): el 68 % de los churners que se escapan tiene renovación automática (el modelo detecta ~5 % de ellos). El 47 % de los churners con renovación automática **cancela después del corte y antes de su vencimiento**: esa información no existe al 31 de enero.

## Siguiente paso propuesto (pendiente de confirmar con el usuario)

**Notebook 10: predicción por ciclo.** Hacer `as-of = vencimiento − k` por cliente, con k ∈ {28, 14, 7, 3, 1}, y graficar la curva AP/F1 contra k (trade-off anticipación vs. precisión). Aplica a w1 y w2 por igual.
- Sin fuga: nada desde el día del vencimiento en adelante.
- Los logs y transacciones de la cohorte de febrero llegan hasta el 2017-02-28; `*_v2` cubre marzo para el test.
- `_common.set_cutoff()` hoy cambia un corte global; para esto hace falta un corte **por cliente**, porque `scan_transactions` y `clean_user_days` asumen un corte único.

Después: commit del EDA → pipeline de w1 (20 features, export v6) → constructor de ciclos de w2 (`_common.payment_cycles` ya existe, sin probar a fondo).

## Cómo ejecutar

- `workshops/.venv` está roto (se creó en otra ruta). Hay que usar el Python del flatpak de VS Code:
  `flatpak run --command=python3 com.visualstudio.code -` (script por stdin), o
  `flatpak run --command=sh com.visualstudio.code -c 'cd <ruta>/workshops/eda && python3 -m nbconvert --to notebook --execute --inplace NB.ipynb'`
- pandas 3 + XGBoost 3.1: las categóricas van como códigos enteros con `feature_types=['c', ...]` (las categorías `str` fallan).
- La 07 lee la caché de la 06 (`cache/06_listening_features.parquet`).
