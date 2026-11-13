# PLAN.md — Proyecto Hugging Face (`hf`)

> Plan maestro. Última actualización: 2026-11-12 (reestructuración a 7 sesiones temáticas tras revisar `GUIA.md`).

## 1. Contexto

- **Tema:** ecosistema Hugging Face (Transformers, Datasets, Hub, Tokenizers, pipelines, fine-tuning).
- **Curso:** CE Inteligencia Artificial y Big Data (Grado E) · **Centro:** I.E.S. Severo Ochoa (Elche) · **Curso:** 2026-2027.
- **Profesor:** José Manuel Pérez Torres.
- **Duración:** 14 h (7 sesiones × 2 h), en paralelo a ML (Manuel Quinto) y DL.
- **Base de diseño:** `GUIA.md` (Manuel Quinto, 29/10/2026), refinada a 2026.

## 2. Normativa (fichas en `FUENTES.md`)

- RD 279/2021 (RA/CE de MIA 5071 y PIA 5073) · RD 497/2024 · Decreto 95/2026 (CV) · Orden 8/2025 (CV, evaluación).

## 3. Estructura final: 7 sesiones temáticas

| # | Sesión | Fecha | Contenido |
|---|---|---|---|
| S28 | Ecosistema y pipelines | 18/11 | Hub, model cards, `pipeline`, elegir modelo |
| S31 | Tokenizadores | 23/11 | BPE/WordPiece/SentencePiece, `AutoTokenizer`, chat templates |
| S33 | Transformers por dentro | 25/11 | atención, 3 familias, logits/softmax, embeddings |
| S36 | La librería Datasets | 30/11 | `load_dataset`, `map`/`filter`, Arrow, publicar |
| S38 | Fine-tuning con Trainer y Evaluate | 02/12 | `Trainer`, `evaluate`, `push_to_hub` |
| S41 | Visión, audio y multimodal | 09/12 | ViT, CLIP, Whisper, VLM |
| S44 | Despliegue: Hub, Gradio, Spaces | 14/12 | `huggingface_hub`, Gradio, Spaces, `InferenceClient` |

> **Numeración alineada con `GUIA.md`** (S28, S31, S33, S36, S38, S41, S44). Datasets (S36) y Fine-tuning (S38) son sesiones **separadas**.

**Actualizaciones a 2026 aplicadas:** familias de Transformers + mención Mamba/SSM (S33), nombres actualizados Whisper/FLUX (S41), Inference Providers (S44). **Agentes (`smolagents`) excluidos** por decisión del docente.

## 4. Relación con los RA (detalle en `docs/relacion_ra.md`)

- **RA1 (MIA):** S28, S44 · **RA2 (MIA):** S28–S44 (central) · **RA3 (MIA):** S28, S33 · **RA5 (MIA):** fuera de alcance · **RA2/RA3 (PIA):** S28, S36, S38, S44.

## 5. Evaluación

Retos semanales (40 %) + Space final (60 %), según la guía de Manuel Quinto.

## 6. Estructura de carpetas

```
hf/
├── PLAN.md, AGENTS.md, README.md, FUENTES.md, GUIA.md, curriculo.yml
├── mkdocs.yml, hooks.py, requirements.txt, serve.sh
└── docs/
    ├── index.md, relacion_ra.md, assets/js/init.js
    ├── sesiones/   (7 sesiones: .md + .ipynb)
    └── evaluacion/ (index.md, retos.md, proyecto_final.md)
```

## 7. Historial de decisiones

- 2026-11-11: esqueleto por módulo/RA; repo y CI a Pages.
- 2026-11-12: 4 bloques por RA redactados.
- 2026-11-12: reestructuración a 7 sesiones temáticas (revisión de `GUIA.md`).

## 8. Pendientes (revisión)

1. ~~Enunciado/CE de los RA.~~ **Hecho** (PIA = 5073; `RAPIA.3` = convergencia tecnológica).
2. ~~Cobertura de RA5.~~ **Decidido:** fuera de alcance (se imparte en `mia`).
3. ~~Evaluación ↔ Orden 8/2025.~~ **Decidido:** retos 40 % + Space 60 %.
4. Calendario/festivos del centro.

## 9. Roadmap · sesiones futuras dedicadas a LLMs

Pendientes de planificar en sesiones posteriores (identificados al revisar *Hugging Face in Action*, Manning 2025):

- **RAG** — consultar datos propios (cap. 12 del libro).
- **LLMs en local** con GPT4All — privacidad y velocidad sin cloud (cap. 11).
- **LangChain / LlamaIndex** — frameworks para aplicaciones LLM (cap. 7).
- **Generación, cuantización y LoRA** — contenido que estaba en la antigua sesión 5 y se aplaza a este bloque de LLMs.

> Fuera de alcance (decisión del docente): agentes (cap. 9) y MCP (cap. 13).
