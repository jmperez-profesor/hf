# Relación con los Resultados de Aprendizaje (RA)

Este curso de Hugging Face se imparte en paralelo a **Machine Learning** (Manuel Quinto) y **Deep Learning** (otro docente). Aunque está organizado por **sesiones temáticas**, se vincula a los Resultados de Aprendizaje oficiales del Curso de Especialización en IA y Big Data.

> Los códigos `RAPIA.3` / `RAMIA.1` y sus criterios se toman del etiquetado oficial de las actividades del material de referencia de **Aitor Medrano** (I.E.S. Severo Ochoa, CC BY-NC-SA 4.0). El enunciado textual de cada RA sigue pendiente de verificar en el RD 279/2021.

## Mapa sesión → RA

| Sesión | Contenido | MIA (5071) | PIA (5072) |
|--------|-----------|------------|------------|
| S1 · Ecosistema y pipelines | Hub, model cards, CLI, `pipeline` | **RA1** (`RAMIA.1`) | **RA3** (`RAPIA.3`) |
| S2 · Tokenizadores | BPE/WordPiece/SentencePiece, `AutoTokenizer` | — | **RA3** |
| S3 · Transformers por dentro | atención, familias, logits, embeddings | — | **RA3** |
| S4 · Datasets y fine-tuning | `datasets`, `Trainer`, `evaluate` | — | **RA3** |
| S5 · LLMs: generación, cuantización, LoRA | generación, bitsandbytes, PEFT | — | **RA3** |
| S6 · Visión, audio y multimodal | ViT, CLIP, Whisper, VLM | — | **RA3** |
| S7 · Despliegue: Hub, Gradio, Spaces | `huggingface_hub`, Gradio, Spaces | **RA1** (`RAMIA.1`) | **RA3** |

## Códigos oficiales identificados

| Código | Módulo | Criterios asociados (según Aitor Medrano) |
|--------|--------|-------------------------------------------|
| **RAMIA.1** | MIA (5071) · RA1 — Caracterización de sistemas de IA | `CEMIA.1d`, `CEMIA.1e` |
| **RAPIA.3** | PIA (5072) · RA3 — Procesamiento de lenguaje natural / programación de IA | `CEPIA.3b`, `CEPIA.3c`, `CEPIA.3d` |

## Detalle de los RA

### MIA (5071) · Modelos de Inteligencia Artificial

- **RA1** (*`RAMIA.1`* — caracteriza sistemas de IA) → Cubierto en **S1** (leer model cards, elegir modelo con criterio) y **S7** (publicar y documentar un sistema).
- **RA2** (*utiliza modelos*) → Cubierto de forma aplicada en **S1–S7** (es el hilo práctico del curso), aunque no es el RA que etiqueta oficialmente el contenido HF.
- **RA5** (*sistemas expertos*) → Se toca **tangencialmente** en S1 (zero-shot). ⚠️ **Revisar:** sin sesión dedicada, la cobertura formal es débil.

### PIA (5072) · Programación de Inteligencia Artificial

- **RA3** (*`RAPIA.3`* — programación / PLN; criterios `CEPIA.3b/c/d`) → **RA principal del curso**: cubierto en **S1–S7**.

> **Nota para revisar:** el etiquetado de Aitor sitúa la mayor parte del contenido de Hugging Face en **PIA · RA3** (no repartido en RA1/RA2/RA5 como se asumió inicialmente). Hay que confirmar el enunciado y los criterios exactos de `RAPIA.3` en el RD 279/2021 y alinear la evaluación con la Orden 8/2025.
