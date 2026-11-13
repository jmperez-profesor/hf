Guía del curso de Hugging Face (7 sesiones × 2 h) 

Oct 29, 2026 · @Manuel Quinto 

# Sesiones:
    - {id: S28, fecha: 2026-11-18, tipo: presencial, contenido: "HF 1: Introducción a la plataforma. El ecosistema y los pipelines "}
    - {id: S31, fecha: 2026-11-23, tipo: presencial, contenido: "HF 2: Spaces y Gradio"}
    - {id: S33, fecha: 2026-11-25, tipo: presencial, contenido: "HF 3: Spaces, Models y Datasets II"}
    - {id: S36, fecha: 2026-11-30, tipo: presencial, contenido: "HF 4: Gradio II"}
    - {id: S38, fecha: 2026-12-02, tipo: presencial, contenido: "HF 5: Tasks I"}
    - {id: S41, fecha: 2026-12-09, tipo: presencial, contenido: "HF 6: Datasets"}
    - {id: S44, fecha: 2026-12-14, tipo: presencial, contenido: "HF 7: "}

# Enfoque 

Curso práctico de 16 h centrado en usar y adaptar modelos preentrenados con el ecosistema Hugging Face, no en construir redes desde cero. Es autocontenido: se puede seguir en paralelo al curso de Deep Learning sin depender de su calendario. 

- Requisitos: Python, pandas y nociones básicas de ML (train/test, métricas). PyTorch no es imprescindible: se usa a nivel de API. 
- No se repite del curso de DL: tensores, autograd, bucles de entrenamiento, optimizadores, CNN/RNN, ni FastAPI/Streamlit (ya vistos en el curso de ML). 
- Sí se trata (porque en DL se ve de forma superficial o más tarde): la intuición de atención y Transformers, solo lo justo para entender qué hace cada modelo. 
- Entorno: Google Colab con GPU T4, cuenta gratuita en el Hub y token de acceso. Fijar versiones en un requirements.txt común (transformers, datasets, evaluate, accelerate, peft, huggingface_hub, gradio). 
- Formato de cada sesión: ~30 min de explicación, ~75 min de notebook guiado, ~15 min de reto corto. 

Pregunta abierta: confirmar con el compañero de DL en qué sesión aparecen atención y Transformers para ajustar la sesión 3. 

# Sesión 28 · El ecosistema y los pipelines 

**Objetivo**: resolver tareas reales con una línea de código y saber elegir un modelo en el Hub. 

- Qué es Hugging Face: Hub (modelos, datasets, Spaces) y librerías principales. 
- Leer una model card: tarea, idioma, licencia, tamaño, métricas y limitaciones. 
- pipeline(): clasificación de texto, NER, resumen, traducción, zero-shot, generación. 
- CPU frente a GPU (device), caché local y autenticación con token. 

Práctica: comparar dos modelos de sentimiento (uno multilingüe y uno en español) sobre 20 frases propias. 
Reto: encontrar en el Hub el mejor modelo para una tarea dada y justificarlo con su model card. 

# Sesión 31 · Tokenizadores 

**Objetivo**: entender cómo el texto se convierte en la entrada numérica de un modelo. 

- Del texto a ids: tokens, subpalabras y vocabulario. 
- Algoritmos: BPE, WordPiece y SentencePiece, con ejemplos en español. 
- AutoTokenizer: input_ids, attention_mask, padding, truncado y tokens especiales. 
- Longitud de contexto y coste: por qué el número de tokens importa. 
- *Chat templates* en modelos conversacionales. 

Práctica: tokenizar el mismo texto con tres modelos y comparar número de tokens y cortes. 
Reto: entrenar un tokenizador pequeño con tokenizers sobre un corpus propio. 

# Sesión 33 · Transformers por dentro 

**Objetivo**: abrir el pipeline y controlar cada paso a mano. 

- Intuición de atención y del bloque Transformer (20 min, sin matemática detallada). 
- Tres familias: *encoder* (BERT), *decoder* (GPT) y *encoder-decoder* (T5) y para qué sirve cada una. 
- AutoModel frente a AutoModelForXxx: cuerpo + cabeza de tarea. 
- AutoConfig, *logits* → probabilidades, id2label. 
- *Embeddings* de frase con sentence-transformers y búsqueda semántica. 

**Práctica**: reproducir a mano el resultado de un pipeline de clasificación (tokenizar, inferir, softmax). 

**Reto**: buscador semántico sobre 100 preguntas frecuentes. 

# Sesión 36 · La librería Datasets 

**Objetivo**: cargar, transformar y publicar datos listos para entrenar. 

- load_dataset desde el Hub y desde CSV/JSON locales; splits y configs. 
- map por lotes, filter, select, train_test_split, class_encode_column. 
- Formato Arrow, caché y streaming para datos grandes. 
- Paso a pandas y a PyTorch (set_format), y DataCollatorWithPadding. 
- Publicar un dataset con su dataset card. 

**Práctica**: preparar un dataset de reseñas en español tokenizado y particionado, que se usará en la sesión 5. 
**Reto**: subir al Hub el dataset FuelConsumptionCo2 del curso de ML con una card completa. 

# Sesión 38 · Fine-tuning con Trainer y Evaluate 

**Objetivo**: ajustar un modelo preentrenado a una tarea propia sin escribir el bucle de entrenamiento. 

- *Transfer learning* aplicado: cuándo basta con zero-shot y cuándo ajustar. 
- TrainingArguments y Trainer: lotes, learning rate, epochs, evaluación y guardado. 
- evaluate: *accuracy*, F1 y compute_metrics. 
- Precisión mixta (fp16) y accelerate para aprovechar la GPU. 
- push_to_hub: publicar el modelo con su card generada. 

**Práctica**: ajustar un modelo encoder en español (p. ej., BETO o RoBERTa-BNE) sobre el dataset de la sesión 4 y compararlo con el zero-shot de la sesión 1. 
**Reto**: ajustar un modelo de NER o de clasificación multietiqueta. 

# Sesión 41 · Más allá del texto: visión, audio y multimodal 

**Objetivo**: aplicar el mismo flujo (procesador → modelo → post-proceso) a otras modalidades. 

- AutoImageProcessor y AutoFeatureExtractor como equivalentes del tokenizador. 
- Visión: clasificación con ViT, detección de objetos y zero-shot con CLIP. 
- Audio: transcripción con Whisper y clasificación de audio. 
- Multimodal: descripción de imágenes y preguntas sobre imágenes. 
- Vistazo a diffusers: generación de imágenes con un modelo ligero. 

**Práctica**: transcribir un audio en español con Whisper y clasificar el texto resultante con el modelo de la sesión 5. 

**Reto**: clasificador de imágenes zero-shot con CLIP para categorías propias. 

# Sesión 44 · Compartir y desplegar: Hub, Gradio y Spaces 

**Objetivo**: convertir un modelo propio en una demo pública y cerrar el curso con un proyecto. 

- huggingface_hub: repositorios, versiones, ramas y modelos privados. 
- Gradio: Interface y Blocks en pocas líneas. 
- Spaces: publicar la demo, hardware gratuito y límites. 
- Inferencia remota con InferenceClient sin descargar el modelo. 
- Buenas prácticas: licencias, sesgos y documentación en la card. 

**Práctica**: publicar en un Space el modelo de la sesión 5 o 6 con una interfaz Gradio. 

**Proyecto final (última hora)**: presentación de 5 min por alumno o pareja de un Space que combine al menos dos sesiones del curso. 

# Evaluación y coordinación con el curso de Deep Learning (DL)

La evaluación se basa en los retos semanales (40 %) y el Space final (60 %). 

Como ambos cursos pueden ir en paralelo, cada sesión incluye lo mínimo que necesita. Si el curso de DL ya lo ha cubierto, ese bloque se reduce a un repaso de 5 min. 

| Sesión HF | Concepto de DL que usa                    | Si aún no se ha visto en DL  |
|-----------|-------------------------------------------|------------------------------|
| 3         | Atención y Transformers                   | Se explica la intuición (20 min) |
| 5         | Fine-tuning, overfitting, learning rate   | Repaso de 10 min con analogía del curso de ML |
| 6         | Parámetros, memoria de GPU                | Se explica con el propio modelo |
| 7         | CNN para imágenes                         | No necesario: ViT y CLIP se usan como cajas negras |
| 8         | Despliegue                                | Gradio sustituye a Streamlit; no se repite FastAPI |

Recursos: [curso oficial de Hugging Face] (https://huggingface.co/learn) y [documentación de las librerías] (https://huggingface.co/docs). 
