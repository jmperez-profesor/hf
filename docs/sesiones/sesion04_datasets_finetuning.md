---
sesion: "04"
titulo: "Datasets y fine-tuning"
fecha: 2026-11-30
duracion: 120 min
---

# Sesión 4 · Datasets y fine-tuning (30/11/2026)

> **Objetivo:** cargar y preparar datos, y ajustar un modelo preentrenado sin escribir el bucle de entrenamiento.
>
> Formato: ~30 min de explicación · ~75 min de notebook guiado · ~15 min de reto.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Cargar datasets del Hub y de CSV/JSON locales (`load_dataset`) |
| 2 | Transformar con `map`, `filter`, `select`, `train_test_split` y `class_encode_column` |
| 3 | Explicar Arrow, caché y streaming, y pasar datos a pandas/PyTorch |
| 4 | Afinar un modelo con `Trainer` + `evaluate` y publicarlo con `push_to_hub` |

## Cuaderno de la sesión

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/sesiones/sesion04_datasets_finetuning.ipynb){ .md-button } · [ver en el sitio](sesion04_datasets_finetuning.ipynb)

## 1 · Cargar datos

```python
from datasets import load_dataset

dataset = load_dataset("imdb")            # desde el Hub
local = load_dataset("csv", data_files="resenas.csv")   # CSV local
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

## 2b · Crear, persistir y publicar datasets

Además de cargar, puedes **crear** un dataset desde estructuras Python y **guardarlo**:

```python
from datasets import Dataset

# desde una lista de diccionarios (p. ej. leídos de una BD o CSV)
datos = [{"texto": "me encanta", "label": 1}, {"texto": "no me gusta", "label": 0}]
ds = Dataset.from_list(datos)          # o from_dict(...)

ds.save_to_disk("mi_dataset")          # Arrow (rápido, comprimido)
ds.to_parquet("mi_dataset.parquet")    # o to_csv / to_json
from datasets import load_from_disk
ds = load_from_disk("mi_dataset")
```

Para **publicarlo en el Hub** (requiere token de escritura):

```python
from huggingface_hub import login
login()                                # pega tu token
ds.push_to_hub("tu-usuario/mi-dataset")
```

## 3 · Fine-tuning con `Trainer`

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate, numpy as np

modelo = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

args = TrainingArguments(
    output_dir="./resultados", evaluation_strategy="epoch",
    num_train_epochs=1, per_device_train_batch_size=16,
    fp16=True,                      # precisión mixta (GPU)
)

accuracy = evaluate.load("accuracy")
def metricas(eval_pred):
    logits, etiquetas = eval_pred
    return accuracy.compute(predictions=np.argmax(logits, -1), references=etiquetas)

trainer = Trainer(model=modelo, args=args,
                  train_dataset=tokenizado["train"], eval_dataset=tokenizado["test"],
                  compute_metrics=metricas)
trainer.train()
trainer.evaluate()
```

`accelerate` + `fp16`/`bf16` aprovechan la GPU. Para **publicar**: `modelo.push_to_hub("usuario/modelo")`.

## Práctica (notebook)

Prepara un **dataset de reseñas en español** (tokenizado y particionado) y **afina un encoder en español** (p. ej. `dccuchile/bert-base-spanish-wwm-uncased` o `BETO`) para clasificación; compáralo con el **zero-shot** de la sesión 1.

## Reto

Afina un modelo de **NER** o de **clasificación multietiqueta**.

> 🔗 Enlaza con el [**proyecto integrador de visión**](../../recursos/proyecto_vision.md): aplica lo aprendido aquí (datasets + `Trainer`) a un modelo de imágenes.

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

**4.** ¿Qué hace `fp16=True` en `TrainingArguments`?

??? tip "Solución"
    Activa precisión mixta en GPU: entrena más rápido y con menos memoria.

**5.** ¿Qué es `push_to_hub`?

??? tip "Solución"
    Sube el modelo (y su card) al Hub para compartirlo y versionarlo.

## 🎬 Vídeos de la sesión

- [Hugging Face: tu puerta de entrada a la IA — Parte 1](https://www.youtube.com/watch?v=49vPqhGO0GY) — DevHack, ~22 min — datasets en Colab: `load_dataset()`, splits, `features`, `map` por batches.
- [Capítulo 5 del LLM Course oficial de HF en español](https://huggingface.co/learn/llm-course/es/chapter5/1) — carga, `map`/`filter`, `train_test_split` y subida al Hub.

## Observaciones

- En Colab gratuito usa subconjuntos (`shuffle` + `select`) y `batch_size` bajo.
- `huggingface-cli login` + token solo para publicar.

## Fuentes y atribución

- Documentación oficial de `datasets`: https://huggingface.co/docs/datasets
- Apartados de crear/persistir/publicar datasets adaptados de **Aitor Medrano** (CC BY-NC-SA 4.0): https://aitor-medrano.github.io/iabd/hf/datasets.html
