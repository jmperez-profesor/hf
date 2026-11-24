# Relación con los Resultados de Aprendizaje (RA)

Este curso de Hugging Face se imparte en paralelo a **ML** (Manuel Quinto) y **Deep Learning**. Aunque está organizado por **sesiones temáticas**, se vincula a los Resultados de Aprendizaje oficiales del CE «Inteligencia Artificial y Big Data».

> **Datos verificados contra el RD 279/2021** (BOE-A-2021-7686, Anexo I). Los códigos `RAPIA.3` / `RAMIA.1` y sus criterios se toman del etiquetado oficial del material de referencia de **Aitor Medrano** (I.E.S. Severo Ochoa, CC BY-NC-SA 4.0).

## Módulos implicados

| Código | Módulo | Rol en este curso |
|--------|--------|-------------------|
| **5071** | Modelos de Inteligencia Artificial (**MIA**) | RA1, RA2 (y tangencialmente RA3/RA5) |
| **5073** | Programación de Inteligencia Artificial (**PIA**) | RA2, RA3 (etiquetado `RAPIA.3` de Aitor) |

> ML es el módulo **5072** (Sistemas de Aprendizaje Automático), que imparte Manuel Quinto en paralelo.

## Mapa sesión → RA

| Sesión | Contenido | MIA (5071) | PIA (5073) |
|--------|-----------|------------|------------|
| S29_31 · Introducción a HF | PLN, Transformer, `pipeline`, Hub | **RA1** · **RA3** (NLP) | **RA3** (convergencia) |
| S34 · Hub: Spaces, Modelos y Datasets | `huggingface_hub`, Gradio, Spaces | **RA1** | **RA2** · **RA3** |
| S36 · La librería Datasets | `datasets`, transformación, publicación | **RA2** | **RA2** (desarrollo) |
| S39 · Fine-tuning y profundización | `Trainer`, `evaluate`, pipelines, tokenizadores, Transformers | **RA2** · **RA3** | **RA2** (desarrollo) |
| S42 · Visión, audio y multimodal | ViT, CLIP, Whisper, VLM | **RA2** | — |

## RA implicados (texto oficial)

### MIA (5071) · Modelos de Inteligencia Artificial

- **RA1** — Caracteriza sistemas de IA relacionándolos con la mejora de la eficiencia operativa. → **S29_31, S34**
- **RA2** — Utiliza modelos de sistemas de IA implementando sistemas de resolución de problemas. → **S29_31–S42** (hilo central)
- **RA3** — Relaciona el PLN con sus aplicaciones determinando su potencial y sus limitaciones. → **S29_31, S39**
- **RA5** — Aplica sistemas expertos. → **Fuera de alcance** de este curso (se imparte en el módulo MIA; ver el proyecto `mia`).

### PIA (5073) · Programación de Inteligencia Artificial

- **RA2** — Desarrolla aplicaciones de IA utilizando entornos de modelado. → **S36, S39, S34**
- **RA3** (`RAPIA.3`) — Evalúa las mejoras en los negocios integrando convergencia tecnológica. → **S29_31, S34** (Hub/Spaces como plataforma que unifica herramientas)

## Evaluación

- **Retos semanales (40 %)** + **Space final (60 %)**, según la guía de Manuel Quinto.
