# AGENTS.md — Proyecto Hugging Face (`hf`)

## What this is

MkDocs Material (tema Material, español) con apuntes y materiales docentes sobre el **ecosistema Hugging Face** (Transformers, Datasets, Hub, Tokenizers, pipelines, fine-tuning) para el Curso de Especialización en IA y Big Data (CE IA&BD), Comunitat Valenciana, curso 2026-2027. No es código de aplicación: es documentación + notebooks.

A diferencia de `mia` (que vive dentro de un único módulo, el 5071), este proyecto es **transversal**: Hugging Face se imparte en **dos módulos** — **MIA (5071)**, RA1/RA2/RA5, y **PIA (5072)**, RA3 —. Por eso la estructura se organiza **por módulo → RA → sesiones** y no solo por RA.

## Commands

```bash
mkdocs serve          # servidor de desarrollo con live reload
mkdocs build          # build estático del sitio
mkdocs build --clean  # build limpio
```

O simplemente `./serve.sh` (crea venv, instala requirements y lanza `mkdocs serve`).

## Setup

```bash
python3 -m venv ~/virtual-envs/hf-mkdocs
source ~/virtual-envs/hf-mkdocs/bin/activate
pip install -r requirements.txt
```

## Key config

- `mkdocs.yml` es la fuente de verdad para nav, tema, plugins y extensiones.
- `use_directory_urls: false` → las URLs usan `.html`.
- **No hay PDF**: solo web + notebooks.
- Render de notebooks vía `mkdocs-jupyter` (`plugins: - mkdocs-jupyter`).
- `hooks.py` inyecta en las páginas generadas desde `.ipynb` dos botones: **Abrir en Colab** y **Descargar .ipynb** (lee `extra.colab` y `extra.raw_base` de `mkdocs.yml`).
- Matemáticas con MathJax; diagramas Mermaid vía `pymdownx.superfences` + `docs/assets/js/init.js`.
- Idioma `es`; admonitions y paleta claro/oscuro (tema en tono `deep orange` para distinguirlo de `mia`).

## Estructura

```
PLAN.md          plan maestro (temporalización por sesiones 2h)
AGENTS.md         este fichero
FUENTES.md        fichas de fuentes (normativa + documentación de HF)
curriculo.yml     datos curriculares (RA/CE, calendario, pesos)
legislacion/      plantilla_sesion.md
docs/             fuente markdown; index, bloques/MIA_*, bloques/PIA_*, evaluacion/, assets/
practicas/        notebooks .ipynb por sesión/bloque
```

## Convenciones de contenido

- Carpeta de bloque: `docs/bloques/<MODULO>_<RA>/`, p. ej. `MIA_RA2/`, `PIA_RA3/`.
- Cada sesión de 2h = un fichero `sesionNN_<tema>.md` siguiendo `legislacion/plantilla_sesion.md`.
- Los notebooks viven en `practicas/` y se enlazan desde su sesión; también puede haber `.ipynb` dentro de `docs/bloques/.../` (como en `mia`) si se prefiere servirlos directamente.
- Citar siempre la **documentación oficial de Hugging Face** como fuente (ver `FUENTES.md`).
- Licencia prevista: CC BY-NC-SA 4.0.

## Historial de decisiones (fechas)

- 2026-11-11: se crea el esqueleto replicando la estructura de `mia` (MkDocs Material + notebooks Colab + gobernanza PLAN/AGENTS/FUENTES), organizado por módulo (MIA y PIA) porque Hugging Face es transversal. Solo esqueleto + plantillas; contenido pendiente.

## Session history

- **Sesión 1 (2026-11-11, modo SCAFFOLD):** creado el esqueleto completo (gobernanza, mkdocs.yml, hooks, plantilla de sesión, índices de bloque y evaluación). Sin contenido de sesiones todavía.

## Pendientes

- Rellenar el enunciado oficial del **RA3 de PIA (5072)** y su desglose de CE en `curriculo.yml` y `FUENTES.md` (verificar contra el RD 279/2021).
- Redactar el contenido de cada sesión y crear sus notebooks.
- Verificar `repo_url` / `extra.colab` / `extra.raw_base` cuando exista el repositorio `jmperez-profesor/hf`.
