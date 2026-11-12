---
sesion: "01"
modulo: MIA
ra: RA1
ce: [4]
fecha_prevista: TBD
duracion: 120 min
titulo: "El Hub como catálogo de sistemas de IA"
---

# Sesión 01 · El Hub como catálogo de sistemas de IA

> **Módulo / RA:** MIA (5071) · **RA1** (*caracteriza sistemas de IA relacionándolos con la mejora de la eficiencia operativa*).
>
> **Criterios de evaluación (CE):** identificar y caracterizar un sistema de IA; valorar su adecuación a un problema de negocio.

## Qué vas a aprender

| # | Al terminar la sesión serás capaz de… |
|---|---|
| 1 | Definir qué es un sistema de IA y qué preguntas lo caracterizan |
| 2 | Buscar y filtrar modelos en el Hub de Hugging Face |
| 3 | Leer una *model card* y extraer sus campos clave |
| 4 | Rellenar una **ficha de caracterización** y decidir si un modelo sirve para un problema |

## Cuaderno de la sesión

El código se trabaja en el cuaderno **`sesion01_caracterizacion.ipynb`**:

[Abrir en Colab](https://colab.research.google.com/github/jmperez-profesor/hf/blob/main/docs/bloques/MIA_RA1/sesion01_caracterizacion.ipynb){ .md-button } · [ver en el sitio](sesion01_caracterizacion.ipynb)

## El caso: auditar modelos antes de adoptarlos

Nuestra empresa quiere adoptar varios sistemas de IA, pero **antes de integrar hay que caracterizar**. Usamos el **Hub** como catálogo y una **ficha** para documentar cada modelo. Solo tras rellenar la ficha se decide si se adopta o se descarta.

## 1 · Las cinco preguntas de la caracterización

Un sistema de IA se caracteriza respondiendo (ver `apuntes.md`):

1. **¿Qué hace?** (tarea)
2. **¿Qué entra y qué sale?**
3. **¿Con qué datos se entrenó?**
4. **¿Cómo se mide?**
5. **¿Qué límites y licencia?**

## 2 · El Hub: buscar y filtrar

En https://huggingface.co/models filtra por **tarea**, **idioma** y **licencia**. El nombre del modelo suele indicar su base y tamaño (`bert-base`, `distilbert`, `*-large`, `*-multilingual-*`).

## 3 · Leer una model card

Una *model card* documenta el modelo (descripción, usos previstos, datos, métricas, limitaciones, licencia). Es la fuente principal para rellenar la ficha.

## 4 · La ficha de caracterización

```
Modelo:  <usuario/modelo>
Tarea:   <qué hace>
Entrada: <tipo>  ->  Salida: <tipo>
Datos:   <dataset(s), idioma>
Métrica: <cómo se evalúa, valor>
Límites: <cuándo NO usarlo>
Licencia:<condiciones>
```

## 5 · De la ficha a la decisión

Una ficha completa te permite decidir si el modelo **mejora la eficiencia operativa** de tu negocio: ¿tarea correcta?, ¿encaja tu flujo?, ¿cubre tu idioma/dominio?, ¿métrica suficiente?, ¿licencia compatible?

## Temporalización (120 min)

- **Apertura (10 min):** qué es un sistema de IA y por qué caracterizar antes de adoptar.
- **Desarrollo (80 min):** búsqueda en el Hub, lectura de model cards, ficha.
- **Cierre (30 min):** práctica propuesta + puesta en común.

## Práctica guiada (con solución)

Caracteriza `nlptown/bert-base-multilingual-uncased-sentiment` rellenando la ficha con la información de su *model card*.

??? tip "Solución"
    **Tarea:** análisis de sentimiento (1–5 estrellas) · **Entrada:** texto → **Salida:** etiqueta + puntuación · **Datos:** reseñas multilingües de Amazon · **Métrica:** accuracy · **Límites:** reseñas cortas de producto, no dominios especializados · **Licencia:** Apache 2.0 (verificar en la card).

## Práctica propuesta (miniproyecto)

- **Enunciado:** elige **2 modelos** del Hub de tareas distintas y rellena su **ficha de caracterización**. Para cada uno, escribe una frase decidiendo si lo adoptarías para un problema concreto y por qué.
- **Entregable:** notebook `sesion01_caracterizacion.ipynb` + las dos fichas.
- **Criterios:** ficha completa (2 p cada una), decisión justificada (1 p).

## Ejercicios de autoevaluación

**1.** Define qué es un "sistema de IA" y por qué no equivale a "un modelo".

??? tip "Solución 1"
    Es el conjunto **datos + modelo + interfaz** que resuelve una tarea. Un modelo aislado no es un sistema: necesita datos con los que se entrenó y una forma de recibir entradas y devolver salidas.

**2.** Enumera las cinco preguntas de la caracterización.

??? tip "Solución 2"
    Qué hace (tarea) · qué entra/sale · con qué datos se entrenó · cómo se mide · qué límites y licencia tiene.

**3.** ¿Qué es una *model card* y por qué hay que leerla antes de usar un modelo?

??? tip "Solución 3"
    La ficha/documentación del modelo (descripción, datos, métricas, limitaciones, licencia). Leerla evita aplicar el modelo fuera de su dominio, idioma o licencia.

**4.** Da dos criterios por los que un modelo con muy buen accuracy puede **no servir** para tu caso.

??? tip "Solución 4"
    Entrenado en otro idioma/dominio (no cubre tus datos) · licencia no comercial · tarea distinta a la tuya · métrica insuficiente para tu tolerancia al error.

**5.** ¿Qué relación hay entre caracterizar un sistema y la eficiencia operativa?

??? tip "Solución 5"
    Caracterizar permite decidir **con criterio** si el sistema encaja y mejora el proceso, evitando adoptar modelos inadecuados (que no aportarían valor o incluso lo empeorarían).

## Observaciones

- El Hub no requiere cuenta para explorar modelos.
- Para probar un modelo solo necesitas `transformers` (`pipeline`), visto en el RA2.
