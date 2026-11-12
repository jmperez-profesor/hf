# PLAN.md — Proyecto Hugging Face (`hf`)

> Plan maestro recuperable entre sesiones. Última actualización: 2026-11-11 (creación del esqueleto).

## 1. Contexto

- **Tema transversal:** ecosistema **Hugging Face** (Transformers, Datasets, Hub, Tokenizers, pipelines, fine-tuning).
- **Curso de especialización:** Inteligencia Artificial y Big Data (Grado E / FP Grado Superior).
- **Comunidad:** Comunitat Valenciana · **Curso académico:** 2026-2027.
- **Profesor:** José Manuel Pérez Torres.
- **Unidad mínima de planificación:** **sesión de 2 horas**.
- **Hilo transversal:** Hugging Face se imparte en **dos módulos** (ver §3).

## 2. Normativa aplicable (resumen; fichas en `FUENTES.md`)

- **RD 279/2021, de 20 abril**: currículo básico; RA/CE de los módulos 5071 (MIA) y 5072 (PIA).
- **RD 497/2024, de 21 mayo**: modifica enseñanzas mínimas de CE.
- **DECRETO 95/2026, de 19 junio** (CV): currículos de los CE de FP en la CV.
- **ORDEN 8/2025, de 22 abril** (CV): evaluación del proceso de enseñanza-aprendizaje.

## 3. Alcance: Hugging Face por módulo y RA

| Módulo | Código | RA | Rol de Hugging Face |
|--------|--------|----|----------------------|
| Modelos de IA (MIA) | 5071 | **RA1** | El Hub como catálogo de sistemas de IA: caracterizar modelos y datasets. |
| Modelos de IA (MIA) | 5071 | **RA2** | **Núcleo:** pipelines, tokenizers, modelos preentrenados, inferencia, fine-tuning. |
| Modelos de IA (MIA) | 5071 | **RA5** | Modelos como componente de sistemas híbridos y guardarraíles de LLM. |
| Programación de IA (PIA) | 5072 | **RA3** | Procesamiento de lenguaje natural con Transformers (clasificación, NER, QA, resumen, generación). |

> El enunciado exacto y los CE del **RA3 de PIA (5072)** están pendientes de verificar contra el RD 279/2021.

## 4. Calendario 26-27 (sesiones de 2h presenciales: lunes y miércoles)

- **Inicio docencia:** 2026-10-01.
- **Vacaciones:** Navidad 2026-12-22 → 2027-01-06 · Pascua 2027-03-25 → 2027-04-05.
- **Festivos que afectan a lun/mié:** pendiente de concretar por centro (mismo calendario que `mia`).

## 5. Detalle de sesiones (contenido y ejercicios)

> **PENDIENTE.** Rellenar sesión a sesión. Por cada sesión usa las subclaves
> **Contenidos detallados**, **Ejercicios y práctica en clase**, **Materiales / Recursos** y
> **Observaciones** (enlaza al notebook `sesionNN_<tema>.ipynb` cuando proceda).

### MIA · RA1 — Caracterización de sistemas de IA
- **Sesión (2 h) · TBD** — el Hub como catálogo de sistemas de IA.
  - Contenidos detallados: _pendiente_.
  - Ejercicios: _pendiente_.

### MIA · RA2 — Utiliza modelos de IA (núcleo) ✅ redactado
- **S01 · Pipelines y modelos preentrenados** — ecosistema HF, Hub, `pipeline`, elegir modelo. *(apuntes + guía + notebook)*
- **S02 · Tokenizers y Datasets** — `AutoTokenizer`, padding/truncation, `datasets`, `.map()`. *(apuntes + guía + notebook)*
- **S03 · Fine-tuning y evaluación** — `Trainer`, métricas de `evaluate`, guardar/subir. *(apuntes + guía + notebook)*
- Hilo conductor del bloque: **SentimentLab** (clasificación de reseñas).

### MIA · RA5 — Sistemas expertos / guardarraíles
- **Sesión (2 h) · TBD** — modelos como componente de sistemas híbridos y guardarraíles.

### PIA · RA3 — Procesamiento de lenguaje natural
- **Sesión (2 h) · TBD** — clasificación de texto y NER.
- **Sesión (2 h) · TBD** — QA, resumen y generación.

## 6. Estructura de carpetas

```
hf/
├── PLAN.md, AGENTS.md, FUENTES.md, README.md, .gitignore, requirements.txt, serve.sh
├── curriculo.yml
├── mkdocs.yml, hooks.py
├── legislacion/
│   └── plantilla_sesion.md
├── practicas/           # notebooks .ipynb por sesión/bloque (pendiente)
└── docs/
    ├── index.md
    ├── assets/js/init.js
    ├── evaluacion/       # bancos de pruebas y rúbricas por RA
    │   ├── index.md, MIA_RA1.md, MIA_RA2.md, MIA_RA5.md, PIA_RA3.md
    └── bloques/
        ├── MIA_RA1/  (index.md + sesiones)
        ├── MIA_RA2/  (index.md + sesiones + notebooks)
        ├── MIA_RA5/  (index.md + sesiones)
        └── PIA_RA3/  (index.md + sesiones + notebooks)
```

## 7. Plantilla de sesión de 2h presenciales (+ trabajo en casa)

Cada `docs/bloques/<MODULO>_<RA>/sesionNN_<tema>.md` sigue `legislacion/plantilla_sesion.md` e incluye:
- **Práctica guiada (con solución):** ejemplo resuelto con código Python (`transformers`, `datasets`, etc.).
- **Práctica propuesta (miniproyecto):** tarea para el alumnado, entregable en un notebook `.ipynb` renderizado por mkdocs-jupyter (con botones Colab/Descargar), con enunciado, entregables y criterios de evaluación.

## 8. Historial de decisiones

- 2026-11-11 (s1): esqueleto replicando `mia`, organizado **por módulo** (MIA + PIA) por ser Hugging Face transversal. Solo esqueleto + plantillas.

## 9. Pendientes

1. Verificar enunciado/CE del RA3 de PIA (5072) en el RD 279/2021.
2. Concretar el calendario/festivos del centro para 26-27.
3. ~~Redactar MIA · RA2 (apuntes + 3 sesiones + 3 notebooks).~~ **Hecho.**
4. Redactar MIA · RA1, MIA · RA5 y PIA · RA3 (apuntes + sesiones + notebooks).
5. ~~Fijar `repo_url` / `extra.colab` / `extra.raw_base`.~~ **Hecho** (repo `jmperez-profesor/hf`).
