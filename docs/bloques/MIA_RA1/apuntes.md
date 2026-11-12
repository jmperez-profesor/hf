# MIA · RA1 — Apuntes: Caracterizar sistemas de IA con Hugging Face

> Material de lectura y consulta del **RA1** (*caracteriza sistemas de IA relacionándolos con la mejora de la eficiencia operativa*). Acompaña a la sesión 01 del bloque.
>
> **Hilo conductor:** antes de integrar cualquier modelo en un negocio, hay que **caracterizarlo**: saber qué hace, con qué datos, cómo se mide y qué límites tiene. El **Hub de Hugging Face** es el catálogo donde hacerlo.

---

## 1 · Qué es un sistema de IA

Un **sistema de IA** no es solo "un modelo": es el conjunto de **datos + modelo + interfaz** que resuelve una tarea. Caracterizarlo es responder, con rigor, a cinco preguntas:

| # | Pregunta | Qué revela |
|---|---|---|
| 1 | ¿Qué hace? | La tarea que resuelve (clasificar, resumir, detectar…) |
| 2 | ¿Qué entra y qué sale? | El tipo de entrada y salida (texto→etiqueta, imagen→texto…) |
| 3 | ¿Con qué datos se entrenó? | Sesgos, idioma, dominio cubierto |
| 4 | ¿Cómo se mide su calidad? | Métricas y benchmarks |
| 5 | ¿Qué límites y licencia tiene? | Cuándo NO usarlo y bajo qué condiciones |

## 2 · El Hub como catálogo de sistemas de IA

El [Hub](https://huggingface.co/models) es un **catálogo de sistemas de IA ya construidos**. Cada entrada es un modelo con:

- **Model card (ficha):** la documentación del modelo.
- **Files:** pesos, `config.json`, tokenizer.
- **Métricas de uso:** descargas, "me gusta".
- **Licencia.**

Buscar bien es la mitad del trabajo. Los filtros clave son **tarea**, **idioma**, **licencia** y **tamaño**.

## 3 · Anatomía de una model card

Una buena *model card* responde a las cinco preguntas de §1. Campos habituales:

| Campo | Qué responde |
|---|---|
| **Model description** | qué hace |
| **Intended uses / out-of-scope** | para qué sí y para qué no |
| **Training data** | con qué se entrenó (y sus sesgos) |
| **Evaluation results** | métricas y benchmarks |
| **Limitations & biases** | fallos y sesgos conocidos |
| **License** | condiciones de uso |

!!! important "La model card es un contrato"
    Usar un modelo sin leer su *model card* es como comprar una máquina sin leer el manual: puedes aplicarlo donde no debe (fuera de su dominio, en otro idioma, con datos que no vió) y obtener resultados engañosos.

## 4 · La ficha de caracterización

Para estandarizar, usa una **ficha** con estos campos (la "ficha de 1 minuto"):

```
Modelo:  <usuario/modelo>
Tarea:   <qué hace>
Entrada: <tipo>  ->  Salida: <tipo>
Datos:   <dataset(s) de entrenamiento, idioma>
Métrica: <cómo se evalúa y con qué valor>
Límites: <cuándo NO usarlo>
Licencia:<condiciones>
```

## 5 · De la caracterización a la decisión (eficiencia operativa)

Caracterizar no es un ejercicio académico: es la base para decidir si un sistema de IA **mejora la eficiencia operativa**. Preguntas de negocio:

- ¿La tarea que hace el modelo es la que yo necesito?
- ¿Su entrada/salida encaja con mi flujo de trabajo?
- ¿Sus datos de entrenamiento cubren mi idioma y dominio?
- ¿Sus métricas son suficientes para mi tolerancia al error?
- ¿Su licencia me permite usarlo comercialmente?

Un modelo con accuracy 0.99 pero entrenado en otro idioma o con licencia no comercial **no sirve** para tu caso, por muy bueno que sea.

## 6 · Ejemplo: caracterizar un modelo

Tomamos `nlptown/bert-base-multilingual-uncased-sentiment`:

- **Tarea:** clasificación de sentimiento (1–5 estrellas).
- **Entrada:** texto → **Salida:** etiqueta + puntuación.
- **Datos:** reseñas multilingües (Amazon), varios idiomas.
- **Métrica:** accuracy sobre reviews.
- **Límites:** pensado para reseñas cortas de producto, no para textos largos o dominios especializados.
- **Licencia:** ver en la card (suele ser Apache 2.0).

Con esa ficha, una tienda como **SentimentLab** sabe si le sirve: sí, para reseñas de producto multilingües.

## Fuentes

- Hub de modelos: https://huggingface.co/models
- Documentación de *model cards*: https://huggingface.co/docs/hub/model-cards
