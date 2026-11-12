---
sesion: "02"
modulo: MIA
ra: RA2
ce: [6]
fecha_prevista: TBD
duracion: 120 min
titulo: "Tokenizers y Datasets"
---

# Sesión 02 · Tokenizers y Datasets

> **Módulo / RA:** MIA (5071) · **RA2** (*utiliza modelos de sistemas de IA implementando sistemas de resolución de problemas*).
>
> **Criterios de evaluación (CE):** explicar cómo se representa el texto para un modelo; preparar un dataset para entrenar.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Explicar por qué un modelo necesita convertir texto en números |
| 2 | Usar `AutoTokenizer` y entender `input_ids`, `attention_mask`, padding y truncation |
| 3 | Cargar y explorar un dataset del Hub con `datasets` |
| 4 | Tokenizar un dataset completo con `.map()` para dejarlo listo para entrenar |

## Cuaderno de la sesión

El código se trabaja en el cuaderno **`sesion02_tokenizers_datasets.ipynb`**:

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/bloques/MIA_RA2/sesion02_tokenizers_datasets.ipynb){ .md-button } · [ver en el sitio](sesion02_tokenizers_datasets.ipynb)

## El caso: preparar los datos de SentimentLab

En la sesión anterior clasificamos con un modelo ya hecho. Ahora nos preparamos para **entrenar uno propio**: hay que entender cómo el modelo "ve" el texto (tokens) y cómo dejar el dataset de reseñas listo.

## 1 · Por qué tokenizar

El modelo no lee texto: lee **números**. El **tokenizador** traduce cada texto a una secuencia de **IDs de token** usando un **vocabulario** fijo:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(tokenizer("Hola, mundo"))
# {'input_ids': [101, 7592, 1010, 2190, 102], 'attention_mask': [1, 1, 1, 1, 1]}
```

- `101` = `[CLS]` (inicio) · `102` = `[SEP]` (fin), en modelos BERT.
- `7592` y `2190` son los tokens de "Hola" y "mundo".

## 2 · Subtokens y vocabulario

Las palabras que no están en el vocabulario se parten en **subtokens**. El prefijo `##` indica que el subtoken continúa la palabra anterior:

```python
print(tokenizer.tokenize("tokenización"))
# ['token', '##ización']
```

!!! note "¿Por qué subtokens?"
    El vocabulario es finito (p. ej. ~30 000 tokens en BERT). Partir palabras permite representar casi cualquier palabra (incluidas las desconocidas o mal escritas) con un número reducido de piezas.

## 3 · `attention_mask`, padding y truncation

- **`attention_mask`:** `1` en tokens reales, `0` en relleno (`[PAD]`), para que el modelo ignore el relleno.
- **Padding:** en *batch*, todas las secuencias se igualan en longitud rellenando con `[PAD]`.
- **Truncation:** las secuencias muy largas se cortan (máx. típico: 512 tokens).

```python
salida = tokenizer("Hola", padding=True, truncation=True, return_tensors="pt")
print(salida)   # tensores PyTorch listos para el modelo
```

## 4 · Cargar un dataset con `datasets`

```python
from datasets import load_dataset

dataset = load_dataset("imdb")   # reseñas de películas, con splits train/test
print(dataset)
# DatasetDict({train: 25000, test: 25000})
print(dataset["train"][0])
# {'text': '...', 'label': 1}
```

`dataset` es un **`DatasetDict`** con los *splits* ya separados. Cada ejemplo es un diccionario con las columnas del dataset.

## 5 · Tokenizar todo el dataset con `.map()`

Aplicamos el tokenizador a **todas** las reseñas de golpe:

```python
def tokenizar(ejemplo):
    return tokenizer(ejemplo["text"], truncation=True, padding="max_length")

tokenizado = dataset.map(tokenizar, batched=True)
print(tokenizado["train"][0].keys())
# dict_keys(['text', 'label', 'input_ids', 'attention_mask'])
```

- `batched=True` procesa por lotes (mucho más rápido).
- Ahora cada ejemplo tiene `input_ids` y `attention_mask` (entrada del modelo) junto a `label` (objetivo).

## 6 · Subconjuntos para ir rápido

Para Colab gratuito, reduce el dataset con `shuffle` + `select`:

```python
pequeno = dataset["train"].shuffle(seed=42).select(range(2000))
```

## Temporalización (120 min)

- **Apertura (10 min):** repasar por qué entrenamos uno propio.
- **Desarrollo (80 min):** tokenizador, padding/truncation, datasets, `.map()`.
- **Cierre (30 min):** práctica propuesta + puesta en común.

## Práctica guiada (con solución)

Dado `tokenizer("Estoy muy contento")`, ¿cuántos `input_ids` esperas y qué significa el primer y el último número?

??? tip "Solución"
    Se generan los tokens de "Estoy", "muy" y "contento" más `[CLS]` al inicio (`101`) y `[SEP]` al final (`102`): en total 5 IDs. (Con un modelo en inglés, algunas palabras españolas se parten en subtokens.)

## Práctica propuesta (miniproyecto)

- **Enunciado:** carga `imdb`, explora 3 ejemplos (texto y etiqueta) y tokeniza un subconjunto de 1000 ejemplos con padding y truncation.
- **Entregable:** notebook `sesion02_tokenizers_datasets.ipynb` ejecutado, mostrando las columnas resultantes.
- **Criterios:** carga correcta (2 p), exploración (1 p), tokenización con `.map()` (2 p).

## Ejercicios de autoevaluación

**1.** ¿Por qué el modelo necesita que el texto se convierta en números?

??? tip "Solución 1"
    Porque una red neuronal opera con tensores numéricos, no con texto. El tokenizador traduce texto a **IDs de token** usando un vocabulario fijo.

**2.** ¿Qué es un subtoken y cuándo aparece?

??? tip "Solución 2"
    Una pieza en que se parte una palabra cuando no está entera en el vocabulario. Se marca con `##` (p. ej. `token` + `##ización`).

**3.** ¿Para qué sirve `attention_mask`?

??? tip "Solución 3"
    Indica qué tokens son reales (`1`) y cuáles son relleno (`0`), para que el modelo ignore el `[PAD]`.

**4.** Diferencia **padding** y **truncation**.

??? tip "Solución 4"
    **Padding** rellena las secuencias cortas para igualar longitudes; **truncation** corta las que superan la longitud máxima.

**5.** ¿Qué hace `.map()` sobre un dataset y por qué usar `batched=True`?

??? tip "Solución 5"
    Aplica una función a todos los ejemplos. `batched=True` procesa por lotes, mucho más rápido que ejemplo a ejemplo.

## Observaciones

- `%pip install datasets transformers` en el Colab.
- `imdb` son ~25 000 ejemplos por split; usa subconjuntos para no esperar.
