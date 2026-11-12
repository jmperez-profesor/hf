---
sesion: "01"
modulo: PIA
ra: RA3
ce: [0]
fecha_prevista: TBD
duracion: 120 min
titulo: "Clasificación de texto y NER"
---

# Sesión 01 · Clasificación de texto y NER

> **Módulo / RA:** PIA (5072) · **RA3** (*procesamiento de lenguaje natural*).
>
> **Criterios de evaluación (CE):** resolver tareas de clasificación de texto y de reconocimiento de entidades con Transformers.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Enumerar las principales tareas de PLN y situar clasificación y NER |
| 2 | Explicar, a nivel conceptual, qué aporta la **atención** de los Transformers |
| 3 | Resolver una tarea de **clasificación de texto** con `pipeline` |
| 4 | Resolver una tarea de **NER** y leer sus entidades |

## Cuaderno de la sesión

El código se trabaja en el cuaderno **`sesion01_clasificacion_ner.ipynb`**:

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/bloques/PIA_RA3/sesion01_clasificacion_ner.ipynb){ .md-button } · [ver en el sitio](sesion01_clasificacion_ner.ipynb)

## El caso: extraer información de reseñas y noticias

SentimentLab recibe reseñas (¿positivas o negativas?) y quiere, además, **detectar automáticamente entidades** (productos, marcas, lugares) en textos de clientes y noticias del sector.

## 1 · El mapa de tareas de PLN

Clasificación, NER, QA, resumen, generación, traducción… (ver `apuntes.md` §2). Hoy nos centramos en las dos primeras.

## 2 · Por qué los Transformers

La **atención** permite que cada palabra se represente teniendo en cuenta **todas las demás** de la frase, en paralelo. Eso captura mejor el contexto (clave para desambiguar) y es la base del PLN moderno.

## 3 · Clasificación de texto

```python
from transformers import pipeline

clasificador = pipeline("sentiment-analysis")
print(clasificador("El pedido llegó roto, fatal."))
```

También hay clasificación por **tema** o **intención** (como el guardarraíl del RA5).

## 4 · NER: entidades token a token

```python
ner = pipeline("ner")
print(ner("Apple abrió una tienda en Valencia el lunes."))
# [{'entity': 'ORG', 'word': 'Apple'}, {'entity': 'LOC', 'word': 'Valencia'}]
```

NER es clasificación **token a token**: a cada palabra se le asigna una etiqueta (persona `PER`, organización `ORG`, lugar `LOC`, …).

## 5 · Reconstruir entidades compuestas

Los modelos suelen devolver los subtokens por separado; con `grouped_entities=True` se agrupan:

```python
ner = pipeline("ner", grouped_entities=True)
```

## Temporalización (120 min)

- **Apertura (10 min):** mapa de tareas de PLN y por qué importan.
- **Desarrollo (80 min):** atención (concepto), clasificación, NER.
- **Cierre (30 min):** práctica propuesta + puesta en común.

## Práctica guiada (con solución)

Sobre `"Tim Cook presentó el nuevo iPhone en Cupertino"`, ¿qué entidades devuelve `ner` y de qué tipo son?

??? tip "Solución"
    Devuelve `Tim Cook` como `PER` (persona), `iPhone` como `MISC`/`PRODUCT` (depende del modelo) y `Cupertino` como `LOC` (lugar). Los tipos exactos dependen del modelo y su esquema de etiquetas.

## Práctica propuesta (miniproyecto)

- **Enunciado:** clasifica el sentimiento de 5 reseñas propias y extrae entidades de una noticia corta (con `grouped_entities=True`).
- **Entregable:** notebook `sesion01_clasificacion_ner.ipynb` ejecutado.
- **Criterios:** clasificación correcta (2 p), NER con entidades agrupadas (2 p), interpretación (1 p).

## Ejercicios de autoevaluación

**1.** Nombra cuatro tareas de PLN e indica su entrada y salida.

??? tip "Solución 1"
    Clasificación (texto→categoría) · NER (texto→entidades) · QA ((contexto,pregunta)→respuesta) · resumen (texto→texto corto) · generación (prompt→texto).

**2.** ¿Qué aporta la **atención** de los Transformers frente a las redes recurrentes?

??? tip "Solución 2"
    Cada token se representa teniendo en cuenta todos los demás de golpe (contexto global) y en paralelo, lo que capta mejor el contexto y entrena más rápido que leer secuencialmente.

**3.** ¿Qué diferencia hay entre clasificación de texto y NER?

??? tip "Solución 3"
    La clasificación asigna **una** categoría a todo el texto; el NER clasifica **token a token**, etiquetando cada palabra/entidad (persona, lugar…).

**4.** ¿Para qué sirve `grouped_entities=True`?

??? tip "Solución 4"
    Para agrupar los subtokens de una misma entidad compuesta (p. ej. "Tim" + "Cook" → "Tim Cook") en lugar de devolverlos por separado.

**5.** ¿Qué es el *transfer learning* en PLN?

??? tip "Solución 5"
    Partir de un modelo **preentrenado** en el lenguaje (BERT, RoBERTa…) y **afinarlo** (fine-tuning) a una tarea concreta con pocos datos.

## Observaciones

- Para español, prueba modelos multilingües o entrenados en español.
- El esquema de etiquetas NER varía según el modelo; lee su *model card*.
