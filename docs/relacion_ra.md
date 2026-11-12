# Relación con los Resultados de Aprendizaje (RA)

Este curso de Hugging Face se imparte en paralelo a **Machine Learning** (Manuel Quinto) y **Deep Learning** (otro docente). Aunque está organizado por **sesiones temáticas**, se vincula a los Resultados de Aprendizaje oficiales del Curso de Especialización en IA y Big Data.

## Mapa sesión → RA

| Sesión | Contenido | MIA (5071) | PIA (5072) |
|--------|-----------|------------|------------|
| S1 · Ecosistema y pipelines | Hub, model cards, `pipeline` | **RA1** (caracterizar sistemas) · **RA2** (usar modelos) | — |
| S2 · Tokenizadores | BPE/WordPiece/SentencePiece, `AutoTokenizer` | **RA2** | — |
| S3 · Transformers por dentro | atención, familias, logits, embeddings | **RA2** | **RA3** (PLN) |
| S4 · Datasets y fine-tuning | `datasets`, `Trainer`, `evaluate` | **RA2** | — |
| S5 · LLMs: generación, cuantización, LoRA | generación, bitsandbytes, PEFT | **RA2** | — |
| S6 · Visión, audio y multimodal | ViT, CLIP, Whisper, VLM | **RA2** | — |
| S7 · Despliegue: Hub, Gradio, Spaces | `huggingface_hub`, Gradio, Spaces | **RA2** | — |

## Detalle de los RA

### MIA (5071) · Modelos de Inteligencia Artificial

- **RA1** — *Caracteriza sistemas de IA relacionándolos con la mejora de la eficiencia operativa.* → Cubierto en **S1** (leer model cards y elegir modelo con criterio).
- **RA2** — *Utiliza modelos de sistemas de IA implementando sistemas de resolución de problemas.* → Cubierto en **S1–S7** (es el RA central del curso).
- **RA5** — *Aplica sistemas expertos evaluando la influencia de los controladores inteligentes.* → Se toca **tangencialmente** en S1 (zero-shot classification como clasificador de intención). ⚠️ **Revisar:** sin una sesión dedicada, la cobertura de RA5 es débil.

### PIA (5072) · Programación de Inteligencia Artificial

- **RA3** — *Procesamiento de lenguaje natural* (enunciado oficial pendiente de verificar contra el RD 279/2021). → Cubierto en **S3** (Transformers/atención) y de forma aplicada en **S1** (pipelines de NLP) y **S5** (generación).

> **Nota para revisar:** el alcance oficial exige alcanzar todos los RA del módulo con evaluación propia (Orden 8/2025). La cobertura de **RA5** y la justificación formal del **RA3 de PIA** quedan pendientes de confirmar.
