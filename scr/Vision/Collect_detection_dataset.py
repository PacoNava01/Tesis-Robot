import cv2
import numpy as np
import csv
from pathlib import Path
from picamera2 import Picamera2

# --- CONFIGURACIÓN ---
OUTPUT_FILE = Path("Jupyter/Tesis-Proyecto/data/dataset.csv")

# Rango de color rojo en HSV
LOW_RED1 = np.array([0, 115, 20])
UP_RED1  = np.array([10, 255, 255])

LOW_RED2 = np.array([170, 100, 35])
UP_RED2  = np.array([180, 255, 255])

# Variables globales
current_label = 1  # 1=objeto, 0=fondo
dataset = []
last_contours = []
last_hsv = None


# --- FUNCIONES ---
def extract_features(contour, hsv_frame):
    """Extrae características geométricas y de color de un contorno."""
    area = cv2.contourArea(contour)
    if area < 10:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    perimeter = cv2.arcLength(contour, True)

    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = (area / hull_area) if hull_area > 0 else 0

    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(hsv_frame, mask=mask)

    return [area, aspect_ratio, circularity, solidity, mean_val[0], mean_val[1], mean_val[2]]


def mouse_callback(event, x, y, flags, param):
    """Captura clics sobre contornos para guardar muestras."""
    global dataset, current_label, last_contours, last_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        for c in last_contours:
            if cv2.pointPolygonTest(c, (x, y), False) >= 0:
                features = extract_features(c, last_hsv)
                if features:
                    dataset.append(features + [current_label])
                    print(f"[+] Etiqueta {current_label} guardada. Total: {len(dataset)}")


def init_cam():
    """Inicializa la cámara PiCamera2."""
    try:
        cam = Picamera2()
        config = cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        cam.configure(config)
        cam.start()
        print("Cámara inicializada correctamente.")
        return cam
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return None


def save_dataset():
    """Guarda el dataset en CSV."""
    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["area", "aspect_ratio", "circularity", "solidity", "h", "s", "v", "label"])
        writer.writerows(dataset)
    print(f"Dataset guardado en {OUTPUT_FILE}")


# --- PROGRAMA PRINCIPAL ---
if __name__ == "__main__":
    cam = init_cam()
    if cam is None:
        exit()

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_callback)

    try:
        while True:
            frame = cam.capture_array()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Máscara roja
            mask1 = cv2.inRange(hsv, LOW_RED1, UP_RED1)
            mask2 = cv2.inRange(hsv, LOW_RED2, UP_RED2)
            mask = cv2.add(mask1, mask2)

            # Limpieza morfológica
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            last_contours, last_hsv = contours, hsv

            display = frame.copy()
            cv2.drawContours(display, [c for c in contours if cv2.contourArea(c) > 400], -1, (0, 255, 0), 1)

            status_txt = f"MODO: {'OBJETO' if current_label == 1 else 'FONDO'} | Muestras: {len(dataset)}"
            cv2.putText(display, status_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Frame", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('o'):
                current_label = 1
            elif key == ord('f'):
                current_label = 0
            elif key == ord('s'):
                save_dataset()
            elif key == 13:  # Enter
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
