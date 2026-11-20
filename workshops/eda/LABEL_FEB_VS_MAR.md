# Cómo se diferencia el churn de febrero del de marzo

Este documento explica qué significa "la regla de marzo" que aparece en la notebook 12 y en `DECISIONS.md`.
Todos los números salen de `12_march_shift_diagnosis.ipynb` (secciones 4 y 5.1) y de la comparación con el script
oficial de KKBox (`WSDMChurnLabeller.scala`).

## Resumen

`train` (febrero) y `train_v2` (marzo) tienen la misma columna, `is_churn`, pero **no la calculan igual**.
Los clientes se comportan igual en los dos meses. Lo que cambia es qué etiqueta recibe un mismo comportamiento.

- En **febrero**, un cliente se queda si hizo **cualquier pago** en los 30 días posteriores a su vencimiento.
- En **marzo**, ese pago no basta: si el pago es de un **plan de más de 31 días**, o si el cliente **paga y luego
  cancela**, se le marca como churn.

Nadie publicó esta diferencia. La deduje comparando las etiquetas de los dos archivos con las transacciones.

## Definiciones

- **Vencimiento:** la fecha de fin de la suscripción conocida en la fecha de corte (31 de enero para febrero, 28 de
  febrero para marzo).
- **Ventana:** los 30 días que van desde el vencimiento hasta vencimiento + 30.
- **Pago normal:** un pago con importe mayor que 0 y un plan de 31 días o menos.
- **Plan largo:** un plan de más de 31 días (90, 100, 180, 195, 410 días, etc.).

Las tablas de este documento se refieren a los clientes cuyo vencimiento cae en el mes de la etiqueta. Son el 89 %
de cada cohorte (883.631 en febrero, 865.793 en marzo). El otro 11 % se trata aparte, más abajo.

## La regla de febrero

La encontró la notebook 11:

> El cliente se queda (`is_churn = 0`) si existe al menos un pago fechado dentro de la ventana.
> Si no existe ninguno, es churn (`is_churn = 1`).

Esta regla reproduce el 99,4 % de las etiquetas de febrero. No mira qué plan se pagó ni qué pasó después del pago.
Tiene una rareza: un pago hecho **antes** del vencimiento no cuenta como renovación.

## Qué cambia en marzo

Según lo que hizo el cliente en la ventana, este es el porcentaje que cada archivo etiqueta como churn:

| Qué hizo el cliente en la ventana | Clientes (feb / mar) | Churn en febrero | Churn en marzo |
|---|---|---|---|
| Pago normal y no canceló después | 818.925 / 713.024 | 0,2 % | 0,1 % |
| **Solo pagos de plan largo** | 12.822 / 10.000 | 2 % | **97 %** |
| **Pagó y después canceló** (vencimiento final a 25 días o menos del original) | 7.962 / 4.586 | 48 % | **96 %** |
| Solo pagos de importe 0 | 1.332 / 937 | 99 % | 82 % |
| Ningún pago | 43.922 / — | ~100 % | no comparable (ver nota) |

Nota: las transacciones disponibles terminan el 31 de marzo. Para un cliente que vence el 20 de marzo solo se ven
11 días de su ventana, así que "ningún pago visto" en marzo no significa que no pagara. Las otras filas no tienen
ese problema: si el pago ya se ve, el caso está decidido.

La primera fila y la última son iguales en los dos meses. Las dos filas en negrita son la diferencia:

### Regla 1: renovar con un plan largo es churn en marzo

Si todos los pagos del cliente dentro de la ventana son de planes largos, febrero lo marca como "se queda" y marzo
como churn. Ejemplo ilustrativo:

| | Vence | Paga | Plan pagado | `is_churn` |
|---|---|---|---|---|
| Cliente de febrero | 10 de febrero | 12 de febrero | 180 días | 0 |
| Cliente de marzo | 10 de marzo | 12 de marzo | 180 días | 1 |

Afecta a unos 10.000 clientes por mes. En marzo son 9.700 churners, el 15 % de todos los churners de marzo.
Casi todos son clientes de pago manual, y la mayoría ya venía de un plan largo.

Si el cliente de plan largo renueva con un plan de 30 días, no es churn en ninguno de los dos meses.

### Regla 2: renovar y luego cancelar es churn en marzo

Si el cliente paga dentro de la ventana y después cancela, quedando con un vencimiento como mucho 25 días posterior
al original, febrero lo marca churn la mitad de las veces y marzo casi siempre. Caso real de marzo:

| Fecha | Registro | Vence |
|---|---|---|
| 1 de febrero | pago, 30 días, renovación automática | 1 de marzo |
| 1 de marzo | pago, 30 días, renovación automática | 1 de abril |
| 9 de marzo | cancelación | 9 de marzo |

Este cliente pagó el mismo día de su vencimiento y tiene `is_churn = 1` en `train_v2`. Con la regla de febrero
habría sido 0. Si después de cancelar vuelve a pagar, marzo lo marca como "se queda".

Afecta sobre todo a clientes de renovación automática.

### En una frase

Febrero pregunta: **¿hubo algún pago en los 30 días tras el vencimiento?**
Marzo parece preguntar: **¿sigue el cliente, tras el vencimiento, con una suscripción mensual activa?**

## Cuánto explica

Los clientes renuevan al mismo ritmo en los dos meses. Por ejemplo, entre los manuales de plan largo, el 44 % paga
en la semana siguiente al vencimiento tanto en febrero como en marzo. Pero su churn pasa de 47 % a 83 %.

Si se aplican las dos reglas a los clientes de febrero, usando sus propias transacciones:

| Segmento | Febrero, etiqueta oficial | Febrero, con reglas 1 + 2 | Marzo, etiqueta oficial |
|---|---|---|---|
| Renovación automática | 1,2 % | 1,9 % | 1,8 % |
| Manual, plan ≤ 31 días | 24,9 % | 29,4 % | 26,8 % |
| Manual, plan > 31 días | 46,8 % | 83,7 % | 83,3 % |
| Cancelación | 46,5 % | 48,5 % | 50,6 % |
| **Todos** | **5,6 %** | **7,4 %** | **7,3 %** |

La diferencia de tasa entre los dos meses (5,6 % contra 7,3 %) queda explicada por la definición de la etiqueta.

## Qué significa "reetiquetar febrero"

Es cambiar el `is_churn` de los clientes de **febrero** antes de entrenar, para que signifique lo mismo que el de
marzo. No se usa ninguna etiqueta de marzo para entrenar ni para elegir el umbral.

```
etiqueta nueva = 1  si la etiqueta oficial de febrero es 1
                    o si todos sus pagos en la ventana son de plan largo        (regla 1)
                    o si pagó en la ventana y luego canceló, con vencimiento
                       final a 25 días o menos del original                     (regla 2)
               = 0  en otro caso
```

- La regla 1 cambia unas 12.600 etiquetas de febrero y la regla 2 unas 4.100.
- La ventana de febrero está cerrada: termina como muy tarde el 30 de marzo y los datos llegan al 31.
- Lo único que vino de marzo es saber cuál es la regla. Por eso su uso está pendiente de aprobación.

Efecto sobre el F1 de marzo, con el mismo modelo, features y protocolo:

| Modelo | Con etiqueta oficial | Con reglas 1 + 2 |
|---|---|---|
| Por ciclo, k = 7 | 0,592 | 0,611 |
| Por ciclo, k = 1 | 0,590 | 0,674 |
| Calendario, cohorte completa | 0,584 | 0,592 |

## El script oficial de KKBox no explica la diferencia

El script `WSDMChurnLabeller.scala` hace esto:

1. Toma como candidatos a los clientes con alguna transacción en el mes anterior cuyo último vencimiento cae en el
   mes de la etiqueta.
2. Busca el primer pago posterior al corte y calcula `gap = fecha del pago − vencimiento`.
3. Marca churn si `gap ≥ 30` o si no hay ningún pago.
4. Una cancelación anterior a ese pago adelanta el vencimiento.

El script no mira nunca la duración del plan ni el importe. Lo porté a Python y lo apliqué a los dos meses:

| | Febrero (`train`) | Marzo (`train_v2`) |
|---|---|---|
| Clientes en el archivo oficial | 992.931 | 970.960 |
| Candidatos según el script | 856.144 | 853.904 |
| En el archivo oficial pero no candidatos | 136.846 | 138.721 |
| Coincidencia de etiqueta en los clientes comunes | 98,8 % | 97,0 % |

Casos en que el script dice "se queda":

| Primer pago tras el corte | Churn oficial en febrero | Churn oficial en marzo |
|---|---|---|
| Pago normal, entre 0 y 29 días tras vencer | 0,5 % | 0,6 % |
| **Plan largo, entre 0 y 29 días tras vencer** | 2,7 % | **93 %** |
| Plan largo, pagado antes de vencer | 95 % | 99 % |
| Pago normal, pagado antes de vencer | 6 % | 10 % |

Conclusiones:

- Ninguno de los dos archivos salió de este script tal cual: los dos incluyen unos 137.000 clientes que el script
  no habría etiquetado.
- Febrero se parece al script en el caso del plan largo. Marzo hace lo contrario. La regla 1 no está en el script.
- El script cuenta un pago anticipado como renovación. Los archivos, en general, no.

La comparación de marzo usa transacciones hasta el 31 de marzo, así que solo es firme donde el script ya dice
"se queda". Las filas de la tabla anterior son de ese tipo.

## Lo que no se sabe

- **La definición real de `train_v2`.** Lo que hay aquí es una descripción que coincide con sus etiquetas, no la
  regla que KKBox aplicó.
- **Por qué un plan largo cuenta como churn.** Como definición de negocio es extraña, pero las etiquetas lo hacen
  con un 97 % de consistencia sobre 10.000 clientes.
- **El 11 % de clientes cuyo vencimiento no cae en el mes de la etiqueta.** Ahí hay más diferencias, solo
  parcialmente identificadas:

| Grupo | Clientes (feb / mar) | Churn en febrero | Churn en marzo |
|---|---|---|---|
| Ya pagados más allá del mes siguiente | 7.574 / 8.214 | 40 % | 90 % |
| Vencen el mes siguiente | 85.485 / 77.433 | 4,0 % | 9,0 % |
| Ya vencidos en el corte | 14.140 / 16.993 | 51 % | 50 % |
| Sin transacciones antes del corte | 2.095 / 2.524 | 32 % | 55 % |

  Tratar como churn al primer grupo (la "regla 3", exploratoria) sube el F1 de marzo del modelo de calendario de
  0,592 a 0,628. Para los que vencen el mes siguiente no encontré la regla: el modelo los sigue subestimando
  (4,3 % predicho contra 9,0 % real).
