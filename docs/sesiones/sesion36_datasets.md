---
sesion: "36"
titulo: "La librería Datasets"
fecha: 2026-11-30
duracion: 120 min
---

# Sesión 36 · La librería Datasets (30/11/2026)

> **Objetivo:** cargar, transformar y publicar datos listos para entrenar.
>
> Formato: ~30 min de explicación · ~75 min de notebook guiado · ~15 min de reto.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Cargar datasets del Hub y de CSV/JSON locales (`load_dataset`) |
| 2 | Transformar con `map`, `filter`, `select`, `train_test_split` y `class_encode_column` |
| 3 | Explicar Arrow, caché y streaming, y pasar datos a pandas/PyTorch |
| 4 | Crear, persistir y publicar un dataset propio en el Hub |

## Cuaderno de la sesión

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/sesiones/sesion36_datasets.ipynb){ .md-button } · [ver en el sitio](sesion36_datasets.ipynb)

## 1 · Cargar datos

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")              # desde el Hub
local = load_dataset("csv", data_files="resenas.csv")   # CSV local
remoto = load_dataset("json", data_files="https://.../datos.json")  # URL
print(dataset)   # DatasetDict con splits train/test
```

## 2 · Transformar

```python
def tokenizar(ejemplo):
    return tokenizer(ejemplo["text"], truncation=True, padding="max_length")

tokenizado = dataset.map(tokenizar, batched=True)         # aplicar a todo
filtrado = dataset.filter(lambda e: len(e["text"]) > 50)  # filtrar
split = dataset["train"].train_test_split(test_size=0.2)  # particionar
```

- **Arrow:** formato columnar eficiente; los datos se cachean en disco.
- **Streaming:** para datasets que no caben en memoria (`load_dataset(..., streaming=True)`), devuelve un `IterableDataset` (se recorre con `next(iter(...))` y se procesa al vuelo).
- **`set_format("torch")`** pasa a tensores; **`DataCollatorWithPadding`** hace padding dinámico por lote.

## 3 · Crear, persistir y publicar

```python
from datasets import Dataset

datos = [{"texto": "me encanta", "label": 1}, {"texto": "no me gusta", "label": 0}]
ds = Dataset.from_list(datos)          # o from_dict(...)

ds.save_to_disk("mi_dataset")          # Arrow (rápido, comprimido)
ds.to_parquet("mi_dataset.parquet")    # o to_csv / to_json
from datasets import load_from_disk
ds = load_from_disk("mi_dataset")
```

Para **publicarlo en el Hub** (token de escritura):

```python
from huggingface_hub import login
login()                                # pega tu token
ds.push_to_hub("tu-usuario/mi-dataset")
```

## Práctica (notebook)

Prepara un **dataset de reseñas en español** tokenizado y particionado, que se usará en la sesión 38 (fine-tuning).

## Reto

Sube al Hub un dataset propio (p. ej. el `FuelConsumptionCo2` del curso de ML) con su *dataset card* completa.

## Ejercicios de autoevaluación

**1.** ¿Qué devuelve `load_dataset("imdb")` y qué son los *splits*?

??? tip "Solución"
    Un `DatasetDict` con los subconjuntos ya separados (p. ej. `train` y `test`).

**2.** ¿Para qué sirve `map(..., batched=True)`?

??? tip "Solución"
    Aplica una función a todos los ejemplos por lotes, mucho más rápido que ejemplo a ejemplo.

**3.** ¿Qué es Arrow y qué aporta?

??? tip "Solución"
    El formato columnar de `datasets`: eficiente en memoria/disco y con caché; junto con streaming permite datos grandes.

## 🎬 Vídeos de la sesión

- [Hugging Face: tu puerta de entrada a la IA — Parte 1](https://www.youtube.com/watch?v=49vPqhGO0GY) — DevHack, ~22 min — datasets en Colab: `load_dataset()`, splits, `features`, `map` por batches.
- [Capítulo 5 del LLM Course oficial de HF en español](https://huggingface.co/learn/llm-course/es/chapter5/1) — carga, `map`/`filter`, `train_test_split` y subida al Hub.

## Observaciones

- En Colab gratuito usa subconjuntos (`shuffle` + `select`) para ir rápido.

## Fuentes y atribución

- Documentación oficial de `datasets`: https://huggingface.co/docs/datasets
- Apartados de crear/persistir/publicar datasets adaptados de **Aitor Medrano** (CC BY-NC-SA 4.0): https://aitor-medrano.github.io/iabd/hf/datasets.html
