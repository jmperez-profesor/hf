# 🤗 Vídeos educativos de Hugging Face y arquitectura Transformer

> Recopilación de vídeos y recursos educativos para el aula — Curso de Especialización en IA y Big Data.
> Elaborado el 12 de noviembre de 2026.

---

## 1. Vídeos en español para introducir Hugging Face en clase

| Vídeo | Canal / Duración | Por qué es útil |
|---|---|---|
| [¡HUGGING FACE DESDE CERO!](https://www.youtube.com/watch?v=29ZGRMlpzu8) | Codificando Bits, ~26 min | Introducción completa y bien estructurada: qué es Hugging Face, modelos, datasets y Spaces, con índice de capítulos por minuto. Perfecto como primer contacto para alumnos. |
| [¿Qué es HUGGING FACE y cómo FUNCIONA?](https://www.youtube.com/watch?v=jG4Ns63tTeo) | 2025 | Explica la plataforma de forma sencilla: repositorio colaborativo, apartados de models, datasets y spaces, y cómo probar demos sin programar. Ideal para alumnos sin experiencia previa. |
| [Playlist: Transformers y LLM con Hugging Face](https://www.youtube.com/playlist?list=PLxJ3eugu174IP9tooyNHzr_bYvasTka3a) | Curso en español, varias entregas | Incluye un vídeo práctico de 13 min sobre la librería Transformers: `pipeline`, tokenizador, clasificación de sentimiento, modelos en español y guardado en local, con notebook de Colab incluido. Ideal para practicar en clase. |
| [Crear imágenes con IA con Hugging Face](https://mediateca.educa.madrid.org/video/p2q2djf3sucn9p9l) | Mediateca EducaMadrid, 2024 | Recurso educativo oficial de una comunidad docente española, en español y pensado para alumnado; se puede incrustar en Moodle con iframe. |

### Recursos oficiales de Hugging Face (con vídeo)

- **Canal oficial de Hugging Face** (<https://www.youtube.com/@HuggingFace/courses>): sección de cursos con grabaciones de sesiones en vivo del LLM Course, Agents Course, Audio Course, etc.
- **Hugging Face Learn** (<https://huggingface.co/learn>): cursos gratuitos y sin anuncios. El **LLM Course está traducido al español**: <https://huggingface.co/learn/llm-course/es>.
- **Education Toolkit** (<https://github.com/huggingface/education-toolkit>): material gratuito para docentes con tutoriales traducidos al español, p. ej. *"Un recorrido por el Hub de Hugging Face"* (20-40 min), slides y vídeo guía donde se explican Transformers, transfer learning y colaboración en el Hub.
- **Hugging Face Classrooms**: espacios colaborativos gratuitos donde docente y alumnos comparten modelos, datasets y demos dentro de un aula virtual.

### Para alumnos avanzados (en inglés)

- [Deploying AI Models with Hugging Face](https://www.youtube.com/watch?v=R8h_gpSpEVU) — freeCodeCamp (6 h 53 min, marzo 2026): curso end-to-end del ecosistema completo (Transformers, audio, Diffusers, vídeo, Gradio y despliegue en Spaces). Excelente como material de ampliación, con capítulos marcados para verlo por bloques.

---

## 2. Cómo funciona por dentro: la serie de DotCSV sobre NLP y Transformers

Referencia en español para entender la **arquitectura Transformer** que hay debajo de todo el ecosistema de Hugging Face. Cuatro vídeos cortos, encadenados y con subtítulos:

| Parte | Vídeo | Duración | Qué explica |
|---|---|---|---|
| 1 | [¡De TEXTO a VECTORES!](https://www.youtube.com/watch?v=Tg1MjMIVArc) | 11 min | Evolución del NLP: de reglas manuales a machine learning, y el salto de palabra a vector. |
| 2 | [¿Qué es un EMBEDDING?](https://www.youtube.com/watch?v=RkYuH_K7Fx4) | 15 min | Word embeddings, vectorización del lenguaje y por qué el espacio vectorial captura significado. |
| 3 | [¿Qué es un TRANSFORMER?](https://www.youtube.com/watch?v=aL-EmKuB078) | 18 min | La joya de la serie: limitaciones de las RNN, mecanismos de atención, vectores Query/Key/Value, producto escalar, matriz de atención y el paper *Attention Is All You Need*. |
| 4 | [¿Por qué estas REDES NEURONALES son tan POTENTES?](https://www.youtube.com/watch?v=xi94v_jl26U) | 15 min | Por qué los Transformers son tan paralelizables (y por eso ganan), y el papel del Positional Encoding. |

> 💡 Si solo hay tiempo para una sesión, la Parte 3 es la más valiosa: explica el mecanismo de atención con el ejemplo de "llave y cerradura" (Query y Key), justo lo que un alumno necesita entender antes de tocar `pipeline()` en Transformers.

---

## 3. Cómo funciona Hugging Face por dentro (la plataforma)

- [Tutorial para usar los modelos de Hugging Face](https://www.youtube.com/watch?v=Puk-AJcnwbA) (2026, ~23 min, español): el mejor vídeo sobre la arquitectura real de la plataforma. Recorre las **cuatro formas de ejecutar un modelo**:
  1. API serverless
  2. Descarga local de modelo pequeño
  3. LLM cuantizado en 4 bits
  4. Servidor de inferencia dedicado con vLLM/TGI

  Incluye notebook de Colab gratuito, auditoría del repositorio de un modelo, gestión de tokens, coste y seguridad (muy útil también para el módulo de cloud).

### Documentación oficial que explica "el interior" de HF

- [¿Cómo los 🤗 Transformers resuelven tareas?](https://huggingface.co/docs/transformers/es/tasks_explained): desglosa qué ocurre dentro de BERT, GPT-2, Wav2Vec2 o ViT al predecir (tokenización, máscaras, cabezas de clasificación).
- [Documentación del Hub](https://huggingface.co/docs/hub/main/en/index): arquitectura técnica del Hub — repositorios Git con versionado, ramas y diffs; más de 2M de modelos, 1.5M de datasets y 1.5M de Spaces; capa de almacenamiento Xet para ficheros grandes.
- [Biblioteca de cliente `huggingface_hub`](https://huggingface.co/docs/huggingface_hub/es/index): cómo interactuar por código con el Hub.
- [Curso de HF en español](https://huggingface.co/docs/course/es/chapter1/1): capítulos 1-4 introducen los conceptos principales de 🤗 Transformers.

---

## 4. Alternativas complementarias en español

- [Arquitectura Transformer: la base revolucionaria](https://canal.uned.es/video/68e8e10e40a56ea69b0e8206) — Canal UNED (2025, 4 min): pieza breve de un MOOC oficial de la UNED, con subtítulos y descarga del vídeo disponible. Útil como cierre o repaso de 5 minutos antes de un examen.
- [Dot CSV Lab](https://www.youtube.com/@DotCSVLab): canal hermano de DotCSV con recortes prácticos y tutoriales extraídos de sus directos de Twitch, útil para sesiones hands-on.
- [Transformers y LLM con Hugging Face](https://iaconclase.com/cursos/transformers-con-hugging-face/) — IA con Clase: curso completo en español (93 lecciones) sobre LLM con Hugging Face.
- [Paso a Paso con Hugging Face, OpenAI, FAISS, Pinecone y PyTorch](https://www.youtube.com/watch?v=pr8-Y4p2sUU) — Robles AI (2025): construcción de un sistema RAG desde cero con código fuente disponible.

---

## 5. Propuesta de secuencia didáctica

| Sesión | Contenido | Duración estimada |
|---|---|---|
| 1 | DotCSV Parte 1 y 2 (introducción al PLN y embeddings) | 30-45 min |
| 2 | DotCSV Parte 3 y 4 (arquitectura Transformer y atención) | 40-60 min |
| 3 | Vídeo práctico de `pipeline` (playlist Transformers y LLM) + notebook Colab | 45 min |
| 4 | "Un recorrido por el Hub de Hugging Face" (Education Toolkit, en español) | 20-40 min |
| 5 | Tutorial de las cuatro formas de ejecutar modelos de HF | 25-30 min |

**Secuencia resumida:** Partes 1-2 de DotCSV como introducción al PLN → Partes 3-4 para la arquitectura Transformer → tutorial de ejecución de modelos de HF como puente hacia la práctica con Transformers y el Hub.

---

*Fuentes: búsquedas web realizadas el 12/11/2026 (YouTube, huggingface.co, github.com, canal.uned.es, mediateca.educa.madrid.org, iaconclase.com).*
