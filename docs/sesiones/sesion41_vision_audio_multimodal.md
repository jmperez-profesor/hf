---
sesion: "41"
titulo: "Visión, audio y multimodal"
fecha: 2026-12-09
duracion: 120 min
---

# Sesión 41 · Visión, audio y multimodal (09/12/2026)

> **Objetivo:** aplicar el mismo flujo (procesador → modelo → post-proceso) a otras modalidades.
>
> Formato: ~30 min de explicación · ~75 min de notebook guiado · ~15 min de reto.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Usar `AutoImageProcessor`/`AutoFeatureExtractor` como equivalentes del tokenizador |
| 2 | Clasificar imágenes (ViT), detectar objetos y hacer zero-shot con CLIP |
| 3 | Transcribir audio con Whisper y clasificar audio |
| 4 | Describir y preguntar sobre imágenes (multimodal), con un vistazo a `diffusers` |

## Cuaderno de la sesión

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/sesiones/sesion41_vision_audio_multimodal.ipynb){ .md-button } · [ver en el sitio](sesion41_vision_audio_multimodal.ipynb)

## 1 · El mismo flujo, otra modalidad

En texto usas un *tokenizer*; en visión/audio un **procesador** (de imagen) o **extractor de características** (de audio). El resto del flujo es idéntico:

```mermaid
flowchart LR
    X[Imagen / Audio] --> P[Procesador<br/>→ tensores] --> M[Modelo] --> R[Postprocesado]
```

## 2 · Visión

```python
from transformers import pipeline

clasificador = pipeline("image-classification")
clasificador("gato.jpg")

# zero-shot con CLIP (categorías propias, sin entrenar)
clip = pipeline("zero-shot-image-classification")
clip("gato.jpg", candidate_labels=["gato", "perro", "coche"])

# detección de objetos
detector = pipeline("object-detection")
```

## 3 · Audio

```python
transcriptor = pipeline("automatic-speech-recognition", model="openai/whisper-small")
transcriptor("audio.wav")

clasificador_audio = pipeline("audio-classification")
```

## 4 · Multimodal y difusión

```python
descriptor = pipeline("image-to-text")          # describir imagen
vqa = pipeline("visual-question-answering")     # preguntar sobre imagen
```

`diffusers` (generación de imágenes) se menciona como vistazo; no es el foco del curso.

## Práctica (notebook)

Transcribe un **audio en español con Whisper** y clasifica el texto resultante con el **modelo afinado de la sesión 38**.

## Reto

Clasificador de imágenes **zero-shot con CLIP** para categorías propias (p. ej. productos de una tienda).

> 🔗 Para un proyecto más completo, haz el [**proyecto integrador de visión**](../../recursos/proyecto_vision.md): afina un ViT con tu dataset y publícalo en el Hub.

## Ejercicios de autoevaluación

**1.** ¿Qué es el equivalente del tokenizador en visión/audio?

??? tip "Solución"
    El **procesador de imagen** (`AutoImageProcessor`) o **extractor de características** (`AutoFeatureExtractor`), que convierten la entrada en tensores.

**2.** ¿Qué hace CLIP en modo zero-shot?

??? tip "Solución"
    Clasifica una imagen en **categorías que tú defines** (texto), sin entrenar, comparando imagen y descripciones.

**3.** ¿Qué tarea hace Whisper?

??? tip "Solución"
    Reconocimiento automático del habla (transcripción de audio a texto), robusto y multilingüe.

**4.** Diferencia `image-to-text` de `visual-question-answering`.

??? tip "Solución"
    `image-to-text` describe la imagen en texto; `visual-question-answering` responde una pregunta concreta sobre la imagen.

**5.** ¿Qué aporta saber usar el mismo flujo (procesador→modelo→post-proceso) en varias modalidades?

??? tip "Solución"
    Reutilizas todo lo aprendido en texto (pipelines, modelos preentrenados, transfer learning) en visión, audio y multimodal.

## 🎬 Vídeos de la sesión

- [Clasificación de imágenes con Vision Transformer](https://www.youtube.com/watch?v=iPQRqNEJiz0) — Codificando Bits, ~27 min — arquitectura ViT, `ViTForImageClassification` y `ViTImageProcessor`.
- [Afinación del Vision Transformer](https://www.youtube.com/watch?v=cbb6Z2VOU9Q) — Codificando Bits, ~50 min — fine-tuning de ViT con `datasets` y `Trainer` (ampliación; une S36 y S41).
- [De audio a texto con Whisper](https://www.youtube.com/watch?v=cwNRJZBhpPs) — 2025 — transcripción (usa el CLI; en clase usamos el pipeline de HF).
- [Audio Course oficial de HF](https://huggingface.co/learn/audio-course) — Whisper desde Transformers (inglés).

## Observaciones

- `%pip install transformers` (para CLIP/Whisper puede hacer falta `torch` y descargar pesos).
- Audio e imágenes: usa archivos pequeños para no saturar la RAM.
