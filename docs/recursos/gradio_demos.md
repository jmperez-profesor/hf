# Demos con Gradio (transformers v5)

> Con [Gradio](https://www.gradio.app/) envuelves cualquier modelo en una **interfaz web**: escribes (o subes una imagen) y ves el resultado, sin tocar código. Estas demos funcionan con **transformers v5** (la última), incluidas las tareas cuyas pipelines se eliminaron.

| Demo | Modelo | Hardware | Tarea en v5 |
|---|---|:---:|---|
| Traductor EN→ES | `Helsinki-NLP/opus-mt-en-es` | 🖥️ CPU | usa `AutoModelForSeq2SeqLM` (no `translation`) |
| Análisis de sentimiento | `distilbert...sst-2` | 🖥️ CPU | `sentiment-analysis` ✅ sigue |
| Reconocimiento de entidades (NER) | por defecto | 🖥️ CPU | `ner` ✅ sigue |
| Resumen | `Qwen/Qwen3-4B-Instruct` | 🔥 GPU | `text-generation` + prompt |
| Pregunta-respuesta | `Qwen/Qwen3-4B-Instruct` | 🔥 GPU | `text-generation` + prompt |
| Describir imagen | `Qwen/Qwen3-VL-4B-Instruct` | 🔥 GPU | `image-text-to-text` (VLM) |

> Los ejemplos de CPU los puedes lanzar en Colab sin GPU o en tu máquina. Los de GPU necesitan la **T4 de Colab**.

---

## 🖥️ CPU

### Traductor EN→ES

```python
%pip install -q transformers gradio
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

modelo = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

def traducir(texto):
    entradas = tokenizer(texto, return_tensors="pt")
    salida = modelo.generate(**entradas)
    return tokenizer.decode(salida[0], skip_special_tokens=True)

gr.Interface(
    fn=traducir, inputs="text", outputs="text",
    title="Traductor EN → ES",
    examples=["I like to play basketball and being a geek"],
).launch()
```

### Análisis de sentimiento

```python
%pip install -q transformers gradio
import gradio as gr
from transformers import pipeline

clasificador = pipeline("sentiment-analysis")

def analizar(texto):
    r = clasificador(texto)[0]
    return f"{r['label']} ({r['score']:.2f})"

gr.Interface(
    fn=analizar, inputs="text", outputs="text",
    title="Análisis de sentimiento",
    examples=["Me encanta este producto", "Llegó roto y tarde"],
).launch()
```

### Reconocimiento de entidades (NER)

```python
import gradio as gr
from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")

def entidades(texto):
    r = ner(texto)
    return "\n".join(f"{e['word']} → {e['entity_group']}" for e in r)

gr.Interface(
    fn=entidades, inputs="text", outputs="text",
    title="Entidades (NER)",
    examples=["Apple abrió una tienda en Valencia."],
).launch()
```

---

## 🔥 GPU (activa T4 en Colab)

### Resumen

```python
%pip install -q transformers gradio
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")

def resumir(texto):
    r = pipe([{"role": "user", "content": f"Resume en 2 frases:\n\n{texto}"}])
    return r[0]["generated_text"][-1]["content"]

gr.Interface(fn=resumir, inputs="text", outputs="text",
             title="Resumen").launch()
```

### Pregunta-respuesta

```python
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")

def responder(contexto, pregunta):
    r = pipe([{"role": "user", "content":
        f"Contexto: {contexto}\n\nPregunta: {pregunta}\nRespuesta:"}])
    return r[0]["generated_text"][-1]["content"]

gr.Interface(
    fn=responder,
    inputs=[gr.Textbox(lines=5, label="Contexto"), gr.Textbox(label="Pregunta")],
    outputs="text",
    title="Pregunta-respuesta",
).launch()
```

### Describir una imagen

```python
import gradio as gr
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="Qwen/Qwen3-VL-4B-Instruct")

def describir(imagen):   # imagen llega como ruta de archivo
    r = pipe([{"role": "user", "content": [
        {"type": "image", "image": imagen},
        {"type": "text", "text": "Describe esta imagen."},
    ]}])
    return r[0]["generated_text"][-1]["content"]

gr.Interface(fn=describir, inputs=gr.Image(type="filepath"), outputs="text",
             title="Describir imagen").launch()
```

---

## Cómo se relaciona con la sesión 44

En la **S29** aprendes a envolver un modelo en Gradio; aquí tienes demos completas para las tareas del curso. En la **S44** las ampliamos (`Blocks`, disposición, componentes) y las **desplegamos a Spaces**.

> Ver también: [Migración transformers v4 → v5](migracion_v5.md) para el detalle de qué pipelines se eliminaron y por qué.
