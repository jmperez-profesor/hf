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
| S1 | Ecosistema y pipelines | 18/11 | Hub, model cards, `pipeline`, elegir modelo |
| S2 | Tokenizadores | 23/11 | BPE/WordPiece/SentencePiece, `AutoTokenizer`, chat templates |
| S3 | Transformers por dentro | 25/11 | atención, 3 familias, logits/softmax, embeddings |
| S4 | Datasets y fine-tuning | 30/11 | `datasets`, `Trainer`, `evaluate`, `push_to_hub` |
| S5 | LLMs: generación, cuantización, LoRA | 02/12 | generación, bitsandbytes, PEFT/LoRA, SFT |
| S6 | Visión, audio y multimodal | 09/12 | ViT, CLIP, Whisper, VLM, diffusers |
| S7 | Despliegue: Hub, Gradio, Spaces | 14/12 | `huggingface_hub`, Gradio, Spaces, `InferenceClient` |

**Fusión aplicada (8 → 7):** Datasets y Fine-tuning en una sola sesión (S4), por ser dos mitades del mismo flujo (preparar datos → entrenar).

**Actualizaciones a 2026 aplicadas:** familias de Transformers + mención Mamba/SSM (S3), DPO/salida estructurada fuera de alcance por decisión (S5), nombres actualizados Whisper/FLUX (S6), Inference Providers (S7). **Agentes (`smolagents`) excluidos** por decisión del docente.

## 4. Relación con los RA (detalle en `docs/relacion_ra.md`)

- **RA1 (MIA):** S1 · **RA2 (MIA):** S1–S7 (central) · **RA5 (MIA):** tangencial (zero-shot, S1) ⚠️ · **RA3 (PIA):** S3 (+ S1, S5 aplicado).

## 5. Evaluación

Retos semanales (40 %) + Space final (60 %). ⚠️ **Pendiente** de reconciliar con la Orden 8/2025 (prueba escrita, escala 1–10, ≥5 por RA).

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
