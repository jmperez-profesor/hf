---
sesion: "01"
modulo: MIA
ra: RA5
ce: [5]
fecha_prevista: TBD
duracion: 120 min
titulo: "Modelos en sistemas híbridos y guardarraíles"
---

# Sesión 01 · Modelos en sistemas híbridos y guardarraíles

> **Módulo / RA:** MIA (5071) · **RA5** (*aplica sistemas expertos evaluando la influencia de los controladores inteligentes en el comportamiento del sistema*).
>
> **Criterios de evaluación (CE):** combinar reglas y modelos; usar un modelo como guardarraíl de un sistema con LLM.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Diferenciar la decisión por **reglas** de la decisión por **modelo** y cuándo combinarlas |
| 2 | Explicar qué es un **sistema híbrido** (neuro-simbólico) |
| 3 | Usar `zero-shot-classification` como **guardarraíl** de intención |
| 4 | Construir un mini-chatbot que une un modelo HF con un motor de reglas |

## Cuaderno de la sesión

El código se trabaja en el cuaderno **`sesion01_guardarrailes.ipynb`**:

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/bloques/MIA_RA5/sesion01_guardarrailes.ipynb){ .md-button } · [ver en el sitio](sesion01_guardarrailes.ipynb)

## El caso: el chatbot de soporte de SentimentLab

SentimentLab quiere un chatbot que atienda clientes. El reto: el chatbot debe **entender** lo que el cliente quiere (modelo) y **decidir** qué hacer **con reglas explicables** (sistema experto). Además, un **guardarraíl** debe impedir que responda fuera de su dominio.

## 1 · Reglas vs. modelos

- **Reglas:** `SI intención = reclamación ENTONCES escalar`. Explicables y auditables.
- **Modelos:** detectan la intención a partir del texto, sin escribir reglas para cada frase.

La solución madura es **combinarlos** (sistema híbrido).

## 2 · El guardarraíl: zero-shot classification

```python
from transformers import pipeline

clasificador = pipeline("zero-shot-classification")

texto = "Quiero devolver un producto que llegó roto"
etiquetas = ["consulta", "reclamacion", "compra", "despedida"]
r = clasificador(texto, etiquetas)
print(r["labels"][0], r["scores"][0])   # 'reclamacion' 0.9...
```

`zero-shot-classification` clasifica en **etiquetas que tú defines**, sin entrenar. Es ideal como guardarraíl porque la salida (etiqueta + confianza) alimenta a las reglas.

## 3 · El motor de reglas encima del modelo

```python
def decidir(texto):
    intencion = clasificador(texto, ["consulta", "reclamacion", "compra"])["labels"][0]
    reglas = {
        "reclamacion": "Escalar a agente humano",
        "compra":       "Ayudar a finalizar la compra",
        "consulta":     "Responder con el FAQ",
    }
    return intencion, reglas[intencion]
```

## 4 · El guardarraíl de dominio

Añadimos una regla: si la confianza es **baja**, el chatbot pide aclaración en vez de decidir (evita errores por frases ambiguas o fuera de dominio):

```python
score = r["scores"][0]
if score < 0.5:
    accion = "Preguntar qué necesita el cliente"
```

## 5 · Sensibilidad y robustez

El **umbral** (`0.5`) equilibra sensibilidad y robustez: muy alto → bloquea demasiado (sensible); muy bajo → deja pasar errores (poco robusto). Ajustarlo es parte del diseño.

## Temporalización (120 min)

- **Apertura (10 min):** repasar sistemas expertos (RA5) y por qué suman modelos.
- **Desarrollo (80 min):** zero-shot, motor de reglas, guardarraíl de dominio.
- **Cierre (30 min):** práctica propuesta + puesta en común.

## Práctica guiada (con solución)

¿Qué devuelve `decidir("Me gustaría comprar otro de los vuestros")`? ¿Y si el score fuera 0.4?

??? tip "Solución"
    Detecta intención `compra` → "Ayudar a finalizar la compra". Si el score fuese 0.4 (baja confianza), el guardarraíl activaría "Preguntar qué necesita el cliente" en lugar de decidir.

## Práctica propuesta (miniproyecto)

- **Enunciado:** amplía el chatbot con **2 intenciones más** (p. ej. `seguimiento` y `queja`) y un guardarraíl de **toxicidad** usando `pipeline("text-classification", model="unitary/toxic-bert")`.
- **Entregable:** notebook `sesion01_guardarrailes.ipynb` ejecutado.
- **Criterios:** intenciones nuevas (2 p), guardarraíl de toxicidad (2 p), reglas coherentes (1 p).

## Ejercicios de autoevaluación

**1.** ¿En qué se diferencia decidir con **reglas** de decidir con un **modelo**?

??? tip "Solución 1"
    Las reglas son explícitas, explicables y auditables (`SI…ENTONCES…`); el modelo aprende de datos y es una caja negra, pero percibe entradas no estructuradas (texto, imagen).

**2.** ¿Qué es un sistema **neuro-simbólico**?

??? tip "Solución 2"
    Un sistema híbrido que combina modelos (percepción) con reglas (decisión explicable), aprovechando lo mejor de ambos.

**3.** ¿Qué es un **guardarraíl** y para qué sirve con un LLM?

??? tip "Solución 3"
    Una capa que valida y acota la salida del modelo antes de ejecutarla. Con un LLM, evita que actúe fuera de dominio, con toxicidad o filtrando datos sensibles.

**4.** ¿Por qué `zero-shot-classification` es útil como guardarraíl?

??? tip "Solución 4"
    Porque clasifica en **etiquetas arbitrarias** sin entrenar, devolviendo etiqueta + confianza que el motor de reglas puede usar como hechos.

**5.** ¿Qué equilibra el **umbral** de confianza de un guardarraíl?

??? tip "Solución 5"
    La **sensibilidad** (cuánto bloquea) frente a la **robustez** (cuánto deja pasar). Un umbral alto bloquea más (más sensible); uno bajo deja pasar más (menos robusto).

## Observaciones

- `zero-shot-classification` puede tardar la primera vez (descarga el modelo).
- El guardarraíl de toxicidad usa un modelo de clasificación binaria del Hub.
