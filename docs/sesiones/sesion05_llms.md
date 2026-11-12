---
sesion: "05"
titulo: "LLMs: generación, cuantización y LoRA"
fecha: 2026-12-02
duracion: 120 min
---

# Sesión 5 · LLMs: generación, cuantización y LoRA (02/12/2026)

> **Objetivo:** usar y adaptar un LLM pequeño con los recursos de Colab.
>
> Formato: ~30 min de explicación · ~75 min de notebook guiado · ~15 min de reto.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Controlar la generación: `temperature`, `top-k`/`top-p`, `max_new_tokens` y chat templates |
| 2 | Explicar la **cuantización** en 8/4 bits (memoria frente a calidad) |
| 3 | Entender **PEFT y LoRA** y por qué se entrena <1 % de los parámetros |
| 4 | Hacer *supervised fine-tuning* con `trl` (`SFTTrainer`) y publicar el adaptador |

## Cuaderno de la sesión

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/sesiones/sesion05_llms.ipynb){ .md-button } · [ver en el sitio](sesion05_llms.ipynb)

## 1 · Controlar la generación

```python
from transformers import pipeline

gen = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
print(gen("Escribe una bienvenida:", max_new_tokens=30, temperature=0.7, top_p=0.9)[0]["generated_text"])
```

- **`temperature`:** más alta = más creativo (y más incoherente); más baja = determinista.
- **`top-k` / `top-p`:** recortan el vocabulario a los tokens más probables.
- **`max_new_tokens`:** longitud de la salida.
- **Chat templates:** formato de roles para modelos conversacionales.

## 2 · Cuantización

Reducir la **precisión** de los pesos (fp16 → int8/int4) para que quepan en menos memoria:

```python
from transformers import BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
```

- **8 bits:** buena calidad, mitad de memoria.
- **4 bits (NF4):** mucho menos memoria, ligera pérdida de calidad.

## 3 · PEFT y LoRA

Fine-tuning completo es caro. **LoRA** (PEFT) congela el modelo y entrena **solo unas matrices pequeñas** (<1 % de parámetros):

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

modelo = prepare_model_for_kbit_training(modelo)
lora = LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.05, task_type="CAUSAL_LM")
modelo = get_peft_model(modelo, lora)
```

## 4 · SFT con `trl`

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=modelo, args=SFTConfig(output_dir="./out", max_seq_length=512),
    train_dataset=dataset,    # dataset de instrucciones (formato chat)
)
trainer.train()
```

Al terminar puedes **fusionar** el adaptador o publicar solo el adaptador (más ligero).

## Práctica (notebook)

Ajusta con **QLoRA** un modelo de <1B parámetros (p. ej. SmolLM2-135M/360M) para que responda con un **estilo o formato fijo**.

## Reto

Mide la diferencia **antes y después** del ajuste con 10 preguntas de prueba.

## Ejercicios de autoevaluación

**1.** ¿Qué efecto tiene subir la `temperature`?

??? tip "Solución"
    Más creatividad/aleatoriedad (y más riesgo de incoherencia); bajarla hace la salida determinista.

**2.** ¿Qué es la cuantización y qué consigue?

??? tip "Solución"
    Reducir la precisión de los pesos (int8/int4) para que el modelo use mucha menos memoria, a cambio de una pérdida mínima de calidad.

**3.** ¿Qué es LoRA y por qué entrena <1 % de parámetros?

??? tip "Solución"
    Congela el modelo y entrena matrices pequeñas añadidas (adaptadores), evitando actualizar todos los pesos.

**4.** ¿Qué es `SFTTrainer`?

??? tip "Solución"
    El entrenador de `trl` para *supervised fine-tuning* de LLM sobre datasets de instrucciones.

**5.** ¿Qué ventaja tiene publicar solo el adaptador en vez del modelo completo?

??? tip "Solución"
    Es mucho más ligero (KB en vez de GB) y se puede compartir/reutilizar sobre el mismo modelo base.

## Observaciones

- `%pip install transformers datasets accelerate peft bitsandbytes trl` en Colab con GPU T4.
- Si no hay GPU, usa SmolLM2-135M y cuantización 8 bits para poder probar.
