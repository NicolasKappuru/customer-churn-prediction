# w5 · Optuna + 5 semillas (TimeGAN vs WTimeGAN v2)

Corrida: 2026-11-26 11:14 → 2026-11-27 00:17. 20/20 entrenamientos sin fallos.

## Protocolo
- Optuna por GAN y por clase: 15 trials, presupuesto reducido (1000 iteraciones; v2 600 conjuntas),
  objetivo = distancia de Fréchet con features del LSTM de w2 (fit vs sintético). Trial 0 = configuración original.
- Luego 5 semillas (0–4) a presupuesto completo (10000 iteraciones) con los mejores parámetros de cada clase.
- Evaluación con la misma lógica del notebook (`EVAL_SEED=110` para submuestrear los reales).
- "Original" = la corrida única de los notebooks (semilla 110, configuración a mano).

## Mejores parámetros (Optuna)
| GAN | label | parámetros |
|---|---|---|
| TimeGAN | 1 y 0 (igual en ambas) | hidden_dim 28, 2 capas, lr 7.8e-4 |
| v2 | 1 | hidden_dim 28, 4 capas, g_lr 1.6e-3, d_lr 6.4e-5, n_critic 3 |
| v2 | 0 | hidden_dim 84, 4 capas, g_lr 3.1e-4, d_lr 3.1e-5, n_critic 3 |

## Resultados (media ± std, 5 semillas)
| label | GAN | Fréchet ↓ | Fréchet original | discriminative ↓ | predictive MAE ↓ | % clasificado en su clase |
|---|---|---|---|---|---|---|
| 1 (churn) | TimeGAN | 0.454 ± 0.078 | 0.964 | 0.485 ± 0.007 | 0.093 ± 0.015 | 82 % |
| 1 (churn) | **v2** | **0.224 ± 0.092** | 0.242 | **0.411 ± 0.054** | 0.093 ± 0.012 | 87 % |
| 0 (stay) | TimeGAN | 0.402 ± 0.238 | 0.087 | 0.494 ± 0.004 | 0.100 ± 0.004 | 30 % |
| 0 (stay) | **v2** | **0.194 ± 0.214** | 0.027 | **0.472 ± 0.014** | 0.092 ± 0.006 | 45 % |

Comparación pareada por semilla (Fréchet): v2 < TimeGAN en 5/5 semillas en label1 y en 4/5 en label0.

## Conclusiones
1. **v2 supera a TimeGAN de forma robusta**: Fréchet ~2× menor en ambas clases, mejor discriminative score,
   y gana en 9 de 10 comparaciones pareadas por semilla. La conclusión del notebook se sostiene con varianza.
2. **Optuna ayudó claramente a TimeGAN en label1** (0.96 → 0.45). En v2 label1 la media (0.22) ≈ el original (0.24):
   la búsqueda no aportó; el original no era una semilla afortunada.
3. **En label0 la búsqueda no se trasladó al presupuesto completo**: las configuraciones de Optuna dan peor
   Fréchet que el original (TimeGAN 0.40 vs 0.09; v2 0.19 vs 0.03) y mucha varianza entre semillas.
   Ganar con 1000 iteraciones no garantiza ganar con 10000 (limitación del presupuesto reducido; decirlo en el informe).
4. **La varianza entre semillas es grande** (p. ej. TimeGAN label0: 0.15–0.71), lo que justifica reportar media ± std
   en vez de una sola corrida.
5. Ningún modelo engaña al discriminador post-hoc (0.41–0.49, ideal 0): hay margen de mejora en realismo.

## Siguiente paso sugerido
Correr 5 semillas de label0 con la **configuración original** (~5 h) para comparar configuración original vs Optuna
con varianza en ambas, en lugar de contra una sola corrida.

## Archivos
- `seed_summary.csv`, `seed_metrics.csv` (una fila por corrida)
- `optuna/<gan>_label<k>_{best_params.json,trials.csv,.db}`
- `seeds/seed<s>/<gan>/label<k>/` (synthetic.npy, checkpoint.pt, history.csv, run.json)
- `archive_label0_with_label1_params/`: 3 corridas de label0 hechas con parámetros de label1 (descartadas)
- Logs: `run_all.log`, `run_all_label0.log`, `run_all_label1.log`, `run_all_part1.log`
