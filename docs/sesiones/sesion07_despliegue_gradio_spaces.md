---
sesion: "07"
titulo: "Desplegar: Hub, Gradio y Spaces"
fecha: 2026-12-14
duracion: 120 min
---

# Sesión 7 · Desplegar: Hub, Gradio y Spaces (14/12/2026)

> **Objetivo:** convertir un modelo propio en una demo pública y cerrar el curso con un proyecto.
>
> Formato: ~30 min de explicación · ~75 min de notebook guiado · ~15 min de proyecto final.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Gestionar repositorios, versiones, ramas y modelos privados con `huggingface_hub` |
| 2 | Construir una demo con Gradio (`Interface` y `Blocks`) |
| 3 | Publicar un **Space** y conocer su hardware y límites |
| 4 | Hacer **inferencia remota** con `InferenceClient` sin descargar el modelo |

## Cuaderno de la sesión

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/sesiones/sesion07_despliegue_gradio_spaces.ipynb){ .md-button } · [ver en el sitio](sesion07_despliegue_gradio_spaces.ipynb)

## 1 · `huggingface_hub`

```python
from huggingface_hub import HfApi, login

login()                       # token de acceso
api = HfApi()
api.create_repo("usuario/mi-modelo", private=False)
```

Gestiona repositorios, **versiones** (commits), **ramas** y **modelos privados/gated**.

## 2 · Gradio en pocas líneas

```python
import gradio as gr
from transformers import pipeline

clasificador = pipeline("sentiment-analysis")

def clasificar(texto):
    return clasificador(texto)[0]["label"]

gr.Interface(fn=clasificar, inputs="text", outputs="text").launch()
```

`Blocks` permite interfaces con varios componentes y estado.

## 3 · Spaces

Un **Space** es un repositorio del Hub con una app (Gradio/Streamlit) que se sirve como web. Hardware gratuito (CPU), con **límites** de uso; GPU es de pago.

## 4 · Inferencia remota con `InferenceClient`

```python
from huggingface_hub import InferenceClient

client = InferenceClient()
print(client.text_classification("Me encanta", model="distilbert-base-uncased-finetuned-sst-2-english"))
```

Usa modelos alojados **sin descargarlos** (serverless; en 2026 también vía *Inference Providers*).

## 5 · Buenas prácticas

Licencias, sesgos conocidos y **documentación en la card**. Antes de publicar: ¿qué licencia tiene el modelo base? ¿qué limitaciones debo declarar?

## Práctica (notebook)

Publica en un **Space** el modelo de la sesión 4 o 5 con una interfaz **Gradio**.

## Proyecto final (última hora)

**Presentación de 5 min** (individual o por pareja) de un Space que **combine al menos dos sesiones** del curso.

## Ejercicios de autoevaluación

**1.** ¿Para qué sirve `huggingface_hub`?

??? tip "Solución"
    Para gestionar repositorios del Hub por código: crear, versionar, ramas y modelos privados/gated.

**2.** ¿Qué diferencia hay entre `gr.Interface` y `gr.Blocks`?

??? tip "Solución"
    `Interface` es la vía rápida (una función → inputs/outputs); `Blocks` permite interfaces más complejas con estado y varios componentes.

**3.** ¿Qué es un Space?

??? tip "Solución"
    Un repositorio del Hub con una app (Gradio/Streamlit) servida como web pública, con hardware gratuito limitado.

**4.** ¿Qué ventaja tiene `InferenceClient`?

??? tip "Solución"
    Permite usar modelos alojados en remoto sin descargarlos, ideal para prototipar y para clientes ligeros.

**5.** ¿Qué debes comprobar antes de publicar un modelo?

??? tip "Solución"
    La licencia del modelo base, sus sesgos/limitaciones, y documentarlos en la model card.

## Observaciones

- `%pip install gradio huggingface_hub` en el Colab.
- Para publicar en Spaces necesitas **token de escritura** (`huggingface-cli login`).
