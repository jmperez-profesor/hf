# AGENTS.md — Proyecto Hugging Face (`hf`)

## What this is

MkDocs Material (español) con un **curso práctico de 14 h (7 sesiones × 2 h)** sobre Hugging Face, para el Curso de Especialización en IA y Big Data (CE IA&BD), Comunitat Valenciana, 2026-2027. No es código de aplicación: es documentación + notebooks.

Se imparte en paralelo a **ML** (Manuel Quinto) y **DL**, sin repetir sus contenidos. La base de diseño es `GUIA.md` (propuesta de Manuel Quinto, 29/10/2026), refinada y actualizada a 2026.

## Commands

```bash
mkdocs serve          # desarrollo con live reload
mkdocs build          # build estático
mkdocs build --clean  # build limpio
```

O `./serve.sh` (crea venv, instala requirements y lanza `mkdocs serve`).

## Setup

```bash
python3 -m venv ~/virtual-envs/hf-mkdocs
source ~/virtual-envs/hf-mkdocs/bin/activate
pip install -r requirements.txt
```

## Key config

- `mkdocs.yml` es la fuente de verdad (nav temático por sesiones).
- `use_directory_urls: false` → URLs `.html`.
- **No hay PDF**: solo web + notebooks.
- Render de notebooks vía `mkdocs-jupyter`; `hooks.py` añade botones **Abrir en Colab** / **Descargar .ipynb**.
- Matemáticas con MathJax; Mermaid vía `pymdownx.superfences` + `docs/assets/js/init.js`.
- Tema `deep orange`, idioma `es`.

## Estructura

```
PLAN.md, AGENTS.md, README.md, FUENTES.md, GUIA.md, curriculo.yml
mkdocs.yml, hooks.py, requirements.txt, serve.sh
docs/
  index.md, relacion_ra.md
  sesiones/    (7 × {sesionNN_*.md, sesionNN_*.ipynb})
  evaluacion/  (index.md, retos.md, proyecto_final.md)
  assets/js/init.js
```

## Convenciones de contenido

- Cada sesión = un `.md` (guía de clase: objetivo, apuntes, práctica, reto, autoevaluación) + un `.ipynb` (notebook Colab con celdas TODO).
- Formato de sesión: ~30 min explicación · ~75 min notebook · ~15 min reto.
- Citar la documentación oficial de HF como fuente (`FUENTES.md`).
- Licencia prevista: CC BY-NC-SA 4.0.

## Decisiones (2026-11-12)

- **7 sesiones temáticas** (no 8): se fusiona Datasets + Fine-tuning en una sesión (S4).
- **Sin agentes (`smolagents`)** por decisión del docente.
- **Evaluación** según la guía (40 % retos + 60 % Space); pendiente de reconciliar con la Orden 8/2025.
- **Nav temático** + página aparte `relacion_ra.md` para la documentación oficial de RA.

## Pendientes (para la revisión)

1. Verificar enunciado/CE del **RA3 de PIA (5072)** (RD 279/2021).
2. Cobertura de **RA5** (sistemas expertos): hoy solo tangencial vía zero-shot en S1.
3. Reconciliar la **evaluación** con la Orden 8/2025 (escala 1–10, ≥5 por RA, prueba escrita).
4. Confirmar fechas/calendario del centro.

## Session history

- **2026-11-11:** esqueleto inicial (estructura por módulo/RA), repo `jmperez-profesor/hf`, CI a Pages.
- **2026-11-12:** redactados los 4 bloques por módulo/RA (7 sesiones + 8 notebooks).
- **2026-11-12:** tras revisar `GUIA.md`, **reestructurado a 7 sesiones temáticas** (nav por sesiones + `relacion_ra.md`), fusionando Datasets+Fine-tuning, sin agentes, con evaluación de la guía.
