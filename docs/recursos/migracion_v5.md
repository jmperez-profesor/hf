# Migración de transformers v4 → v5

> `transformers` **v5** (finales de 2025) eliminó varias pipelines "de texto a texto" y de visión. Este curso **fija `transformers<5`** para seguir usando esas pipelines sencillas. Esta página documenta el **antes (v4)** y el **después (v5)** por si usas la última versión.

## Qué cambió

La v5 elimina estas tareas de `pipeline()`:

| Tarea eliminada en v5 | Se usaba en… |
|---|---|
| `translation` / `translation_xx_to_yy` | S29 |
| `summarization` | S29 |
| `question-answering` | S34 |
| `text2text-generation` | — |
| `image-to-text` | S42 |
| `visual-question-answering` | S42 |
| `image-to-image` | — |

**Motivo oficial:** Hugging Face recomienda usar un **chat model** con `text-generation` (mayor calidad) en lugar de pipelines especializadas. Las pipelines "básicas" (`sentiment-analysis`, `ner`, `zero-shot-classification`, `text-generation`, `automatic-speech-recognition`…) **siguen funcionando** en v5.

---

## Antes y después (por tarea)

### 🔁 Traducción

**Antes (v4):**
```python
from transformers import pipeline
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
print(pipe("I like to play basketball")[0]["translation_text"])
```

**Después (v5) — opción A · modelo + tokenizer directos (funciona en CPU):**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

modelo = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

entradas = tokenizer("I like to play basketball", return_tensors="pt")
print(tokenizer.decode(modelo.generate(**entradas)[0], skip_special_tokens=True))
```

**Después (v5) — opción B · chat model (necesita GPU):**
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")
print(pipe([{"role": "user", "content": "Translate to Spanish: I like to play basketball"}])[0]["generated_text"])
```

### 📝 Resumen

**Antes (v4):**
```python
pipe = pipeline("summarization", model="facebook/bart-large-cnn")
print(pipe("...texto largo...", max_length=30)[0]["summary_text"])
```

**Después (v5) — chat model:**
```python
pipe = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")
print(pipe([{"role": "user", "content": "Summarize:\n\n...texto largo..."}])[0]["generated_text"])
```

### ❓ Question answering

**Antes (v4):**
```python
pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
print(pipe(question="¿Dónde está?", context="...")["answer"])
```

**Después (v5) — chat model:**
```python
pipe = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")
prompt = "Context: ...\n\nQuestion: ¿Dónde está?\nAnswer:"
print(pipe(prompt)[0]["generated_text"])
```

### 🖼️ Visión: describir y preguntar sobre imágenes

**Antes (v4):**
```python
pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
print(pipe("gato.jpg")[0]["generated_text"])
```

**Después (v5) — VLM (vision-language model):**
```python
pipe = pipeline("image-text-to-text", model="Qwen/Qwen3-VL-4B-Instruct")
mensaje = [{"role": "user", "content": [
    {"type": "image", "image": "gato.jpg"},
    {"type": "text", "text": "Describe esta imagen."},
]}]
print(pipe(mensaje)[0]["generated_text"])
```

---

## Cómo lo maneja este curso

| Opción | Qué es | Estado |
|---|---|---|
| **1 · Fijar v4** | `course-requirements.txt` fija `transformers<5`; los notebooks funcionan tal cual | ✅ **activa** |
| **2 · Migrar a v5** | adaptar los ejemplos con la tabla de arriba (o un chat model) | 📚 documentada aquí |

> 🎨 ¿Quieres usar **v5 de forma sencilla para el alumnado**? Envuelve los modelos en **Gradio**: así no importa qué cambie la API. Ver [Demos con Gradio (v5)](gradio_demos.md).

## Cuadernos de ejemplo

- [Cuaderno v4 · Pipelines clásicas](cuaderno_v4_pipelines.ipynb) — los ejemplos tal como se vieron en clase (v4, sin Gradio).
- [Cuaderno v5 · Ejemplos con Gradio y Hub](cuaderno_v5_gradio.ipynb) — los mismos ejemplos en v5, con Gradio y el Hub.

!!! tip "Si usas v5 y un ejemplo falla"
    O fijas `transformers<5` en tu entorno (`pip install "transformers<5"`), o adaptas la tarea eliminada con la tabla de arriba. Los notebooks del curso instalan con `%pip install "transformers<5"` para evitar el problema.

> Fuente: [MIGRATION_GUIDE_V5.md](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) (sección *Pipelines*).
