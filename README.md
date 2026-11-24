# Aplicación YOLO: radar de velocidad con multa simulada

Proyecto de **Redes Neuronales** (Universidad de Sonora).
Autor: Carlos Enrique Mancillas Duran

Una página web que detecta vehículos con **YOLOv8n**, mide qué tan rápido van y, si pasan el límite, genera una **multa simulada** con su placa. Funciona desde el celular y todo corre en el navegador, sin servidor.

**Demo:** https://effulgent-paletas-5010d1.netlify.app

> Es un proyecto académico. La velocidad es una estimación y la multa no tiene validez oficial.

## ¿Cómo funciona?
1. **YOLOv8n** encuentra los vehículos y las placas en cada cuadro del video.
2. Un **seguimiento** (por IoU) le da un número a cada auto para saber que es el mismo de un cuadro a otro.
3. Hay dos líneas, **A y B**, con una distancia real conocida. Se mide el tiempo entre que el auto cruza una y otra:
   `velocidad = distancia ÷ tiempo`
4. Si pasa el límite, se recorta la placa, un **OCR (PP-OCRv4)** la lee y se arma el ticket.

## Parámetros de la app
| Parámetro | Para qué sirve |
|---|---|
| Límite de velocidad | Arriba de este valor hay multa |
| Distancia A–B (m) | Metros reales entre las dos líneas |
| Líneas A y B | Dónde se empieza y termina de medir |
| Dirección | Si los autos van de frente/atrás o cruzan de lado |
| Confianza mínima | Qué tan seguro debe estar YOLO para aceptar una detección |

## Entrenamiento
- Dataset: Roboflow *License Plates* (1,084 imágenes, 416×416, clases `license-plate` y `vehicle`).
- Se entrenó en el clúster **La Yuca** (GPU AMD Instinct MI210) con transferencia de aprendizaje desde YOLOv8n.
- Resultados en el conjunto de prueba: **mAP50 = 0.83**, precisión 0.92, recall 0.78, mAP50-95 0.66.
- Las gráficas de pérdida y exactitud y las matrices de confusión están en `resultados/`.

## Estructura
```
app/             página web (index.html, app.js) y modelos .onnx
entrenamiento/   entrenar_yolo.py, entrenar_yolo.slrm y el ambiente de conda (yolo_multas.yml)
resultados/      gráficas, matrices de confusión y métricas
```

## Cómo probarlo en tu compu
Los modelos se cargan con `fetch`, así que hay que abrir la carpeta `app/` con un servidor local (no sirve abrir el HTML con doble clic):
```bash
cd app
python -m http.server 8000
```
Luego abre http://localhost:8000, sube un video o usa la cámara y pon los parámetros.

## Limitaciones
- La distancia A–B es una estimación: si está mal, todas las velocidades salen escaladas igual.
- El dataset tiene más autos vistos por detrás, así que de frente detecta peor.
- Con poca luz, autos muy lejanos o tapados por otros, el auto puede no medirse.
- El OCR puede equivocarse con placas borrosas (el campo de la placa se puede corregir a mano).
