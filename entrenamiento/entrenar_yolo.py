#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tarea 7 - Aplicación de YOLO (Redes Neuronales, UNISON)
=======================================================
Entrena un detector YOLOv8 de VEHÍCULOS y PLACAS en el cluster La Yuca.
Al terminar:
  * genera las gráficas de "accuracy" (precision, recall, mAP) y de loss,
  * copia la matriz de confusión,
  * evalúa en el conjunto de prueba,
  * exporta el modelo a ONNX + meta.json para la app web del celular.

Uso
---
1) En el NODO DE LOGIN (tiene internet), una sola vez:
       python entrenar_yolo.py --preparar
   (descarga los pesos yolov8n.pt y la fuente que usa Ultralytics para graficar,
    porque los nodos de cómputo normalmente no tienen internet)

2) Mandar a la cola con SLURM:
       sbatch entrenar_yolo.slrm
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin pantalla en el cluster
import matplotlib.pyplot as plt
import pandas as pd
import yaml


# ----------------------------------------------------------------------------
# Argumentos
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento YOLO vehículos + placas")
    p.add_argument("--data", default="dataset/data.yaml",
                   help="data.yaml del dataset (formato YOLOv8 de Roboflow)")
    p.add_argument("--modelo", default="yolov8n.pt",
                   help="pesos preentrenados de partida (yolov8n.pt es el más ligero)")
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu", help="'cpu' o '0' si usas la partición gpu")
    p.add_argument("--paciencia", type=int, default=20, help="early stopping")
    p.add_argument("--nombre", default="multas_yolo", help="nombre del experimento")
    p.add_argument("--salida", default="salida", help="carpeta con la evidencia final")
    p.add_argument("--preparar", action="store_true",
                   help="solo descarga pesos/fuentes (correr en nodo de login)")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def preparar(modelo):
    """Descarga todo lo que necesita internet, para que el job corra offline."""
    from ultralytics import YOLO
    from ultralytics.utils.checks import check_font
    YOLO(modelo)                 # descarga yolov8n.pt en la carpeta actual
    check_font("Arial.ttf")      # fuente para las gráficas de Ultralytics
    try:
        check_font("Arial.Unicode.ttf")
    except Exception:
        pass
    print("Listo: pesos y fuentes descargados. Ya puedes hacer: sbatch entrenar_yolo.slrm")


def arreglar_data_yaml(ruta_yaml):
    """Roboflow exporta rutas relativas tipo '../train/images' que a veces
    Ultralytics no resuelve bien. Se reescribe un data.yaml con rutas absolutas."""
    ruta_yaml = Path(ruta_yaml).resolve()
    raiz = ruta_yaml.parent
    with open(ruta_yaml) as f:
        cfg = yaml.safe_load(f)

    def buscar(*opciones):
        for o in opciones:
            if (raiz / o).exists():
                return str((raiz / o).resolve())
        return None

    nuevo = {
        "path": str(raiz),
        "train": buscar("train/images", "images/train"),
        "val": buscar("valid/images", "val/images", "images/val"),
        "test": buscar("test/images", "images/test"),
        "names": cfg["names"],
    }
    if nuevo["val"] is None:              # si no hay validación, usar train
        nuevo["val"] = nuevo["train"]
    if nuevo["test"] is None:
        nuevo.pop("test")
    if nuevo["train"] is None:
        sys.exit(f"No encontré train/images dentro de {raiz}")

    salida = raiz / "data_abs.yaml"
    with open(salida, "w") as f:
        yaml.safe_dump(nuevo, f, allow_unicode=True)
    print("data.yaml corregido ->", salida)
    print(yaml.safe_dump(nuevo, allow_unicode=True))
    return str(salida), nuevo


def graficar(results_csv, carpeta):
    """Gráficas de loss y de 'accuracy' a partir de results.csv.
    En detección no existe 'accuracy' como tal: se usan precision, recall,
    mAP@0.5 y mAP@0.5:0.95, que son las métricas equivalentes."""
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    ep = df["epoch"]

    # ---- Loss ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, nombre, titulo in zip(
        axs, ["box_loss", "cls_loss", "dfl_loss"],
        ["Box loss (localización)", "Cls loss (clasificación)", "DFL loss"]):
        if f"train/{nombre}" in df:
            ax.plot(ep, df[f"train/{nombre}"], label="entrenamiento", lw=2)
        if f"val/{nombre}" in df:
            ax.plot(ep, df[f"val/{nombre}"], label="validación", lw=2, ls="--")
        ax.set_title(titulo); ax.set_xlabel("época"); ax.set_ylabel("loss")
        ax.grid(alpha=.3); ax.legend()
    fig.suptitle("YOLOv8 - Pérdidas (loss)", fontsize=14)
    fig.tight_layout()
    fig.savefig(Path(carpeta) / "grafica_loss.png", dpi=150)
    plt.close(fig)

    # ---- "Accuracy" (métricas) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    etiquetas = {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP@0.5",
        "metrics/mAP50-95(B)": "mAP@0.5:0.95",
    }
    for col, et in etiquetas.items():
        if col in df:
            ax.plot(ep, df[col], label=et, lw=2)
    ax.set_ylim(0, 1); ax.set_xlabel("época"); ax.set_ylabel("valor")
    ax.set_title("YOLOv8 - Precisión / mAP en validación")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(Path(carpeta) / "grafica_accuracy.png", dpi=150)
    plt.close(fig)
    print("Gráficas guardadas en", carpeta)


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.preparar:
        preparar(args.modelo)
        return

    # Ultralytics: no intentar conectarse a internet desde el nodo de cómputo
    os.environ.setdefault("YOLO_OFFLINE", "1")

    import torch
    from ultralytics import YOLO

    hilos = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    torch.set_num_threads(hilos)
    print(f"PyTorch {torch.__version__} | hilos CPU: {hilos} | "
          f"CUDA disponible: {torch.cuda.is_available()}")

    data_yaml, cfg = arreglar_data_yaml(args.data)
    nombres = cfg["names"]
    if isinstance(nombres, list):
        nombres = {i: n for i, n in enumerate(nombres)}
    print("Clases:", nombres)

    salida = Path(args.salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)

    # ------------------------- Entrenamiento -------------------------
    t0 = time.time()
    modelo = YOLO(args.modelo)
    modelo.train(
        data=data_yaml,
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=min(8, hilos),
        patience=args.paciencia,
        project=str(Path("runs").resolve()),
        name=args.nombre,
        exist_ok=True,
        plots=True,          # genera confusion_matrix.png, results.png, etc.
        verbose=True,
        cache="ram",
    )
    print(f"Entrenamiento terminado en {(time.time() - t0) / 60:.1f} min")

    run_dir = Path("runs").resolve() / args.nombre
    best = run_dir / "weights" / "best.pt"

    # ------------------------- Gráficas -------------------------
    graficar(run_dir / "results.csv", salida)
    for archivo in ["confusion_matrix.png", "confusion_matrix_normalized.png",
                    "results.png", "PR_curve.png", "F1_curve.png",
                    "val_batch0_pred.jpg", "results.csv"]:
        if (run_dir / archivo).exists():
            shutil.copy(run_dir / archivo, salida / archivo)

    # ------------------------- Evaluación en test -------------------------
    modelo_best = YOLO(str(best))
    metricas = {}
    split = "test" if "test" in cfg else "val"
    m = modelo_best.val(data=data_yaml, split=split, imgsz=args.imgsz,
                        device=args.device, project=str(Path("runs").resolve()),
                        name=args.nombre + "_" + split, exist_ok=True, plots=True)
    metricas = {
        "split": split,
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
    }
    cm_test = Path("runs").resolve() / (args.nombre + "_" + split) / "confusion_matrix.png"
    if cm_test.exists():
        shutil.copy(cm_test, salida / f"confusion_matrix_{split}.png")
    print("Métricas finales:", json.dumps(metricas, indent=2))
    with open(salida / "metricas.json", "w") as f:
        json.dump(metricas, f, indent=2)

    # ------------------------- Exportar para la app -------------------------
    ruta_onnx = modelo_best.export(format="onnx", imgsz=args.imgsz,
                                   opset=12, simplify=False, dynamic=False)
    app_dir = salida / "model"
    app_dir.mkdir(exist_ok=True)
    shutil.copy(ruta_onnx, app_dir / "best.onnx")
    shutil.copy(best, salida / "best.pt")
    with open(app_dir / "meta.json", "w") as f:
        json.dump({"names": {int(k): v for k, v in nombres.items()},
                   "imgsz": args.imgsz}, f, indent=2, ensure_ascii=False)

    print("\n================ LISTO ================")
    print("Evidencia en:", salida)
    print("Copia la carpeta", app_dir, "dentro de la carpeta app/ antes de subir a Netlify")


if __name__ == "__main__":
    main()
