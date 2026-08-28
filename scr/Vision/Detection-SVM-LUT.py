import cv2
import numpy as np
from picamera2 import Picamera2
from pathlib import Path
import time
import json

# Ruta de la LUT
LUT_PATH = "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/src/Vision/ArUcocalib/color_lut.npy"

# Cargar LUT
lut_matrix = np.load(LUT_PATH)

def init_cam():
    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    print("Cámara inicializada correctamente.")
    return cam

def main():
    camara = init_cam()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    try:
        while True:
            frame_raw = camara.capture_array()
            if frame_raw is None:
                break

            # Rotar frame si es necesario
            frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_180)

            # Convertir a HSV
            hsv_frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2HSV)

            # Indexación directa con LUT
            mask = lut_matrix[hsv_frame[:,:,0], hsv_frame[:,:,1], hsv_frame[:,:,2]] * 255

            # Limpieza morfológica
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

            # Superponer detección
            overlay = frame_raw.copy()
            overlay[mask_clean > 0] = [0,255,0]

            cv2.imshow("Detección LUT", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camara.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
