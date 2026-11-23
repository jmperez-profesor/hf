# demo_v5.py — demo sencilla de análisis de sentimiento con transformers v5 + Gradio.
# Pensada para probar en el DevContainer (que usa transformers v5).
#
# Cómo usarla:
#   1. Abre el repositorio en VS Code y pulsa "Reopen in Container".
#   2. Ejecuta:  python3 demo_v5.py
#   3. Abre la URL que imprime (http://127.0.0.1:7860) y escribe una reseña.

import gradio as gr
from transformers import pipeline

clasificador = pipeline("sentiment-analysis")


def analizar(texto):
    r = clasificador(texto)[0]
    return f"{r['label']} ({r['score']:.2f})"


gr.Interface(
    fn=analizar,
    inputs="text",
    outputs="text",
    title="Análisis de sentimiento (v5)",
    examples=["Me encanta este producto", "Llegó roto y tarde"],
).launch()
