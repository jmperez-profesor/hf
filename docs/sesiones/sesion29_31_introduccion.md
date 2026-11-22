---
sesion: "29_31"
titulo: "Introducción a la plataforma Hugging Face"
fecha: 2026-11-18
duracion: 240 min
---

# Sesiones 29 y 31 · Introducción a la plataforma Hugging Face (18 y 23/11/2026)

> Sesiones introductorias que combinan **PLN**, la **arquitectura Transformer** y **Hugging Face**, siguiendo la presentación del curso.
>
> 📄 [Descargar la presentación](../assets/pdf/IABD_HF_SESIONES_29_y_31.pdf)

## Qué vas a aprender

| # | Al terminar serás capaz de… |
|---|---|
| 1 | Explicar qué es el PLN y su evolución (de reglas a *deep learning*) |
| 2 | Entender la arquitectura Transformer (atención, QKV, multi-head, ventana de contexto) |
| 3 | Usar `pipeline()` y el Hub de Hugging Face |

## 1 · Procesamiento del Lenguaje Natural (PLN)

- El PLN busca que una máquina **interprete y produzca** el idioma como un humano.
- **Georgetown (1954):** primer experimento; tradujo 60 frases ruso→inglés con 250 palabras y 6 reglas. Demostró que el lenguaje **no se puede atrapar con reglas manuales**.
- **Tres eras del PLN:** reglas (1950-80) → estadística (1990-2010) → *deep learning* / Transformer (2017-hoy). En lugar de decirle a la máquina *cómo* hablamos, la dejamos **descubrir cómo hablamos a partir de los datos**.
- El lenguaje está lleno de **ambigüedad** (sintáctica, léxica, semántica, pragmática…): *"Me quedé esperándote en el banco"* (¿asiento o entidad?).
- La comprensión tiene **4 capas**: morfología, sintaxis, semántica y pragmática.

## 2 · La arquitectura Transformer

- Los **embeddings** convierten palabras en vectores: palabras con significado parecido quedan **cerca** en el espacio.
- La **tokenización** divide el texto en tokens y los convierte en números.
- La **atención** (auto-atención) hace que cada palabra "mire" a las demás para deducir su contexto: *"El gato no cruzó la calle porque estaba cansado"* → el modelo sabe que *estaba* se refiere a *gato*.
- **QKV** (Query, Key, Value) y **multi-head attention** son los mecanismos internos.
- La **ventana de contexto** es el límite de memoria del modelo: de 1 frase (Georgetown, 1954) a **millones de tokens** (Claude 4.6, GPT-5.2, marzo 2026).
- Dos familias: **encoder** (BERT, entiende) y **decoder** (GPT, genera).

## 3 · Hugging Face

- Plataforma que **democratiza** los modelos Transformer: el **Hub** (modelos, datasets, Spaces) + las librerías `transformers` (texto) y `diffusers` (imágenes).
- **`pipeline()`** abstrae los pasos (preprocesar → inferir → postprocesar) para usar un modelo en pocas líneas.
- Tareas con una sola función: sentimiento, NER, traducción, resumen, generación, question answering.
- **Hub:** cuenta, token de acceso (lectura/escritura), `hf auth login`, `hf cache scan`.

## Práctica · Hola Traductor y Hola Difusor

- **Traductor** (`Helsinki-NLP/opus-mt-en-es`) y **generación de imágenes** (Stable Diffusion), en Colab o en local.

## Ejercicios (sesión 31)

### Actividad 1 · Traductor + Stable Diffusion (3p)
Crea un entorno virtual e instala las librerías necesarias para ejecutar el **traductor** y el modelo **Stable Diffusion**. Genera un par de frases e imágenes, anota en comentarios el tiempo de generación y adjunta las traducciones y las imágenes generadas.

### Actividad 2 · Cuenta, CLI y caché (1p)
Crea una cuenta en Hugging Face, instala el CLI, haz login (`hf auth login`) y averigua cuánto ocupan los modelos descargados (`hf cache scan`).

## Profundización

Esta sesión introductoria se profundiza en la [**S34 · Transformers, pipelines y tokenizadores**](sesion34_transformers.md).
