import cv2
import numpy as np
from ultralytics import YOLO
from database import guardar_captura


class DetectorAnimalesAI:

    def __init__(self, modelo_path="yolov8n.pt"):
        self.model = YOLO(modelo_path)

    def procesar_detecciones(
            self, results, imagen_original, guardar_crops=False
    ):
        conteo = {"perros": 0, "gatos": 0}
        crops_guardados = []

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            nombre_clase = self.model.names.get(cls_id, "").lower()

            # Identificar si es perro (16) o gato (15) en COCO o por nombre
            if cls_id == 16 or "dog" in nombre_clase or "perro" in nombre_clase:
                conteo["perros"] += 1
                etiqueta = "perro"
            elif cls_id == 15 or "cat" in nombre_clase or "gato" in nombre_clase:
                conteo["gatos"] += 1
                etiqueta = "gato"
            else:
                continue

            # Si está activado, recortar y guardar la foto de la mascota detectada
            if guardar_crops:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = (
                    int(coords[0]),
                    int(coords[1]),
                    int(coords[2]),
                    int(coords[3]),
                )
                h, w, _ = imagen_original.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    crop = imagen_original[y1:y2, x1:x2]
                    ruta = guardar_captura(crop, etiqueta, conf)
                    crops_guardados.append(
                        {"etiqueta": etiqueta, "confianza": conf, "ruta": ruta}
                    )

        return conteo, crops_guardados

    def procesar_imagen(
            self, imagen_pil, confianza=0.5, solo_perros_gatos=True, guardar_crops=False
    ):
        classes = [15, 16] if solo_perros_gatos else None
        img_cv2 = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)

        results = self.model(img_cv2, conf=confianza, classes=classes)
        annotated_frame = results[0].plot()
        imagen_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        conteo, crops = self.procesar_detecciones(
            results, img_cv2, guardar_crops=guardar_crops
        )
        return imagen_rgb, conteo, crops

    def procesar_frame(
            self, frame, confianza=0.5, solo_perros_gatos=True, guardar_crops=False
    ):
        classes = [15, 16] if solo_perros_gatos else None
        results = self.model(frame, conf=confianza, classes=classes)
        annotated_frame = results[0].plot()
        imagen_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        conteo, crops = self.procesar_detecciones(
            results, frame, guardar_crops=guardar_crops
        )
        return imagen_rgb, conteo, crops