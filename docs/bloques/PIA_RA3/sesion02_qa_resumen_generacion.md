---
sesion: "02"
modulo: PIA
ra: RA3
ce: [0]
fecha_prevista: TBD
duracion: 120 min
titulo: "Question answering, resumen y generación"
---

# Sesión 02 · QA, resumen y generación

> **Módulo / RA:** PIA (5072) · **RA3** (*procesamiento de lenguaje natural*).
>
> **Criterios de evaluación (CE):** resolver tareas de question answering, resumen y generación con Transformers.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Diferenciar QA **extractiva** de generativa |
| 2 | Resolver una tarea de **question answering** sobre un contexto |
| 3 | **Resumir** un texto con `summarization` |
| 4 | **Generar** texto con `text-generation` y controlar su longitud |

## Cuaderno de la sesión

El código se trabaja en el cuaderno **`sesion02_qa_resumen_generacion.ipynb`**:

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/bloques/PIA_RA3/sesion02_qa_resumen_generacion.ipynb){ .md-button } · [ver en el sitio](sesion02_qa_resumen_generacion.ipynb)

## El caso: atender y resumir para SentimentLab

SentimentLab quiere un asistente que **responda preguntas sobre sus FAQs** (QA), que **resuma** hilos de soporte largos y que **genere** respuestas amables.

## 1 · Question answering (extractiva)

Dado un **contexto** y una **pregunta**, el modelo **extrae** la respuesta del propio contexto (no la inventa):

```python
from transformers import pipeline

qa = pipeline("question-answering")
qa(
    question="¿Cuál es el horario de atención?",
    context="Nuestro horario de atención al cliente es de 9 a 18 horas, de lunes a viernes.",
)
# {'answer': 'de 9 a 18 horas', ...}
```

!!! note "Extractiva vs. generativa"
    La QA **extractiva** copia un fragmento del contexto; la **generativa** puede redactar una respuesta nueva. La mayoría de modelos del Hub para `question-answering` son extractivos.

## 2 · Resumen

```python
resumidor = pipeline("summarization")
resumidor("...texto largo...", max_length=30, min_length=10)
```

`max_length`/`min_length` controlan la longitud del resumen (en tokens).

## 3 · Generación de texto

```python
gen = pipeline("text-generation")
gen("Escribe una bienvenida para una tienda online:", max_new_tokens=30)
```

`max_new_tokens` limita cuántos tokens **nuevos** genera el modelo.

## 4 · Controlar la salida

- **`max_new_tokens`** / **`max_length`:** longitud.
- **`num_return_sequences`:** cuántas variantes devolver.
- **`do_sample`** + **`temperature`:** más o menos aleatoriedad (más temperatura = más creativo, menos predecible).

## Temporalización (120 min)

- **Apertura (10 min):** repasar las tres tareas y sus entradas/salidas.
- **Desarrollo (80 min):** QA, resumen, generación y parámetros.
- **Cierre (30 min):** práctica propuesta + puesta en común.

## Práctica guiada (con solución)

¿Qué diferencia hay entre `max_new_tokens` y `max_length` en generación?

??? tip "Solución"
    `max_new_tokens` limita los tokens **nuevos** generados; `max_length` limita el total (prompt + generación). En `text-generation` se suele usar `max_new_tokens`.

## Práctica propuesta (miniproyecto)

- **Enunciado:** escribe un contexto de FAQ de SentimentLab (3–4 frases) y formula 2 preguntas para QA; resume un hilo de soporte; y genera 2 respuestas de bienvenida variando `temperature`.
- **Entregable:** notebook `sesion02_qa_resumen_generacion.ipynb` ejecutado.
- **Criterios:** QA correcto (2 p), resumen (2 p), generación con parámetros (1 p).

## Ejercicios de autoevaluación

**1.** Diferencia QA extractiva y generativa.

??? tip "Solución 1"
    La **extractiva** copia un fragmento del contexto; la **generativa** redacta una respuesta nueva. La mayoría de modelos del Hub para QA son extractivos.

**2.** ¿Qué parámetro controla la longitud de un resumen?

??? tip "Solución 2"
    `max_length` y `min_length` (en tokens) acotan la longitud del resumen.

**3.** ¿Qué hace `max_new_tokens` en generación?

??? tip "Solución 3"
    Limita cuántos tokens **nuevos** genera el modelo (sin contar el prompt).

**4.** ¿Qué efecto tiene subir la `temperature`?

??? tip "Solución 4"
    Aumenta la aleatoriedad/creatividad de la salida (y el riesgo de incoherencias). Bajarla la hace más predecible y determinista.

**5.** ¿Por qué la QA extractiva "no se inventa" respuestas, a diferencia de la generación?

??? tip "Solución 5"
    Porque se limita a señalar un fragmento del contexto dado; no produce texto nuevo, así que no puede alucinar contenido fuera del contexto.

## Observaciones

- `summarization` y `text-generation` pueden ser lentos en CPU; usa textos cortos.
- Los modelos generativos pueden **alucinar** (inventar datos); verifica siempre su salida (relaciona con el guardarraíl del RA5).
