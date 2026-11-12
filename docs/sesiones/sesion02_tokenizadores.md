---
sesion: "02"
titulo: "Tokenizadores"
fecha: 2026-11-23
duracion: 120 min
---

# Sesión 2 · Tokenizadores (23/11/2026)

> **Objetivo:** entender cómo el texto se convierte en la entrada numérica de un modelo.
>
> Formato: ~30 min de explicación · ~75 min de notebook guiado · ~15 min de reto.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Explicar por qué un modelo necesita que el texto se convierta en números |
| 2 | Distinguir los algoritmos **BPE, WordPiece y SentencePiece** |
| 3 | Usar `AutoTokenizer`: `input_ids`, `attention_mask`, padding, truncación y tokens especiales |
| 4 | Entender la **longitud de contexto** y por qué el número de tokens importa |

## Cuaderno de la sesión

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/sesiones/sesion02_tokenizadores.ipynb){ .md-button } · [ver en el sitio](sesion02_tokenizadores.ipynb)

## 1 · Del texto a ids

El modelo no lee texto: lee **números**. El tokenizador traduce cada texto a una secuencia de **IDs de token** usando un **vocabulario** fijo:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(tokenizer("Hola, mundo"))
# {'input_ids': [101, 7592, 1010, 2190, 102], 'attention_mask': [1, 1, 1, 1, 1]}
```

`101` = `[CLS]` (inicio) · `102` = `[SEP]` (fin), en modelos BERT.

## 2 · Subpalabras y algoritmos

Las palabras fuera del vocabulario se parten en **subpalabras**. Tres algoritmos dominan:

| Algoritmo | Idea | Uso típico |
|---|---|---|
| **BPE** | Fusiona los pares más frecuentes | GPT, RoBERTa |
| **WordPiece** | Maximiza probabilidad del corpus | BERT, DistilBERT |
| **SentencePiece** | Trata el texto como flujo (sin espacios) | T5, Llama, modelos multilingües |

```python
print(tokenizer.tokenize("tokenización"))
# ['token', '##ización']   # '##' = continúa la palabra anterior
```

## 3 · `attention_mask`, padding y truncación

- **`attention_mask`:** `1` en tokens reales, `0` en relleno, para ignorar el `[PAD]`.
- **Padding:** igualar longitudes en *batch* rellenando.
- **Truncación:** cortar lo que supere la longitud máxima.

```python
salida = tokenizer("Hola", padding=True, truncation=True, return_tensors="pt")
```

## 4 · Longitud de contexto y coste

Los modelos tienen un **límite de tokens** (p. ej. 512 en BERT, miles en LLM modernos). Más tokens = **más memoria y más coste** de inferencia/entrenamiento. Por eso medir en tokens, no en palabras, es esencial.

## 5 · Chat templates

Los modelos conversacionales esperan un **formato de chat** (roles `system`/`user`/`assistant`). El *chat template* del tokenizador lo genera:

```python
mensajes = [{"role": "user", "content": "Hola, ¿qué horario tenéis?"}]
print(tokenizer.apply_chat_template(mensajes, tokenize=False))
```

## Práctica (notebook)

Tokeniza el **mismo texto con tres modelos** (p. ej. BERT, GPT-2 y T5) y compara número de tokens y cortes.

## Reto

Entrena un **tokenizador pequeño con `tokenizers`** sobre un corpus propio (unas frases) y observa el vocabulario generado.

## Ejercicios de autoevaluación

**1.** ¿Por qué el texto debe convertirse en números?

??? tip "Solución"
    Porque la red neuronal opera con tensores numéricos; el tokenizador traduce texto a IDs de token con un vocabulario fijo.

**2.** Diferencia BPE, WordPiece y SentencePiece en una frase.

??? tip "Solución"
    **BPE** fusiona pares frecuentes · **WordPiece** maximiza probabilidad del corpus · **SentencePiece** trata el texto como flujo sin espacios (multilingüe).

**3.** ¿Qué es una subpalabra y cuándo aparece?

??? tip "Solución"
    Una pieza en que se parte una palabra ausente del vocabulario, marcada con `##` (p. ej. `token` + `##ización`).

**4.** ¿Para qué sirven `attention_mask` y la truncación?

??? tip "Solución"
    `attention_mask` marca tokens reales (`1`) vs. relleno (`0`); la truncación corta secuencias que superan la longitud máxima.

**5.** ¿Qué es un *chat template*?

??? tip "Solución"
    El formato de conversación (roles system/user/assistant) que el tokenizador aplica para modelos conversacionales.

## Observaciones

- `%pip install transformers` en el Colab.
- Para el reto: `%pip install tokenizers`.
