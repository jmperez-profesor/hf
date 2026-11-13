---
sesion: "38"
titulo: "Fine-tuning con Trainer y Evaluate"
fecha: 2026-12-02
duracion: 120 min
---

# Sesión 38 · Fine-tuning con Trainer y Evaluate (02/12/2026)

> **Objetivo:** ajustar un modelo preentrenado a una tarea propia sin escribir el bucle de entrenamiento.
>
> Formato: ~30 min de explicación · ~75 min de notebook guiado · ~15 min de reto.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Explicar el *transfer learning*: cuándo basta con zero-shot y cuándo ajustar |
| 2 | Montar un `Trainer` con `TrainingArguments` y una métrica de `evaluate` |
| 3 | Afinar un modelo de clasificación y evaluarlo (accuracy/F1) |
| 4 | Aprovechar la GPU (fp16/`accelerate`) y publicar con `push_to_hub` |

## Cuaderno de la sesión

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/sesiones/sesion38_finetuning.ipynb){ .md-button } · [ver en el sitio](sesion38_finetuning.ipynb)

## 1 · Transfer learning aplicado

El **fine-tuning** reentrena un modelo preentrenado durante pocas épocas con tus datos. Regla práctica: para tareas genéricas basta con **zero-shot** (visto en S28); cuando tu dominio es específico, **ajusta** el modelo.

## 2 · `Trainer` y `evaluate`

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

`accelerate` + `fp16`/`bf16` aprovechan la GPU. Añade `evaluate.load("f1")` para clases desbalanceadas.

## 3 · Publicar

```python
modelo.push_to_hub("tu-usuario/mi-modelo")     # genera la model card
tokenizer.push_to_hub("tu-usuario/mi-modelo")
```

## Práctica (notebook)

Ajusta un modelo **encoder en español** (p. ej. `dccuchile/bert-base-spanish-wwm-uncased` o BETO) sobre el dataset de la sesión 36 y compáralo con el **zero-shot** de la sesión 28.

## Reto

Afina un modelo de **NER** (`AutoModelForTokenClassification`) o de **clasificación multietiqueta**.

> 🔗 Enlaza con el [**proyecto integrador de visión**](../../recursos/proyecto_vision.md): aplica lo aprendido aquí (`Trainer`) a un modelo de imágenes.

## Ejercicios de autoevaluación

**1.** ¿Qué es el fine-tuning y qué ventaja tiene frente a entrenar desde cero?

??? tip "Solución"
    Reentrenar poco tiempo un modelo preentrenado con tus datos: menos datos, menos tiempo y mejores resultados.

**2.** ¿Qué hace `fp16=True` en `TrainingArguments`?

??? tip "Solución"
    Activa precisión mixta en GPU: entrena más rápido y con menos memoria.

**3.** ¿Qué es `push_to_hub`?

??? tip "Solución"
    Sube el modelo (y su card) al Hub para compartirlo y versionarlo.

**4.** ¿Por qué el accuracy puede engañar con clases desbalanceadas?

??? tip "Solución"
    Predecir siempre la clase mayoritaria ya da un accuracy alto sin aprender nada; conviene mirar F1 y la matriz de confusión.

## Observaciones

- En Colab gratuito usa subconjuntos y `batch_size` bajo; `huggingface-cli login` + token solo para publicar.

## Fuentes y atribución

- Documentación de `Trainer`: https://huggingface.co/docs/transformers/training
