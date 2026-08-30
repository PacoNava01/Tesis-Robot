import cv2
import numpy as np
from picamera2 import Picamera2
from pathlib import Path
import time
import json

# Ruta de la LUT
LUT_PATH = "/home/pacon/Tesis-Robot/scr/Vision/data_calib/color_lut.npy"

# Cargar LUT
lut_matrix = np.load(LUT_PATH)

def init_cam() -> Picamera2 | None:
    """Inicializa y configura la cámara Picamera2."""
    try:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        print("Cámara inicializada correctamente.")
        return cam
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return None

def Preprocess(frame,rotate = True,k = (5,5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    if rotate == True:
        # Rotar frame si es necesario
        frame_raw = cv2.rotate(frame, cv2.ROTATE_180)
    else: pass

    # --- CORRECCIÓN DE CANAL PARA LA LUT ---
    # Si el LUT fue calibrada con BGR, descomenta la siguiente línea 
    # para intercambiar los canales rojo y azul antes de pasar a HSV:
    frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
    # Convertir a HSV
    hsv_frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2HSV)

    # Indexación directa con LUT
    mask = lut_matrix[hsv_frame[:,:,0], hsv_frame[:,:,1], hsv_frame[:,:,2]]
    mask = (mask * 255).astype(np.uint8)

    return mask

def main():
    camara = init_cam()
    if camara is None:
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    window_name = "Detección LUT"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frame_raw = camara.capture_array()
            if frame_raw is None:
                break

            # Rotar frame si es necesario
            frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_180)

            # --- CORRECCIÓN DE CANAL PARA LA LUT ---
            # Si tu LUT fue calibrada con BGR, descomenta la siguiente línea 
            # para intercambiar los canales rojo y azul antes de pasar a HSV:
            frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
            
            # Convertir a HSV
            hsv_frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2HSV)

            # Indexación directa con LUT
            mask = lut_matrix[hsv_frame[:,:,0], hsv_frame[:,:,1], hsv_frame[:,:,2]]
            mask = (mask * 255).astype(np.uint8)

            # Limpieza morfológica
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

            # Superponer detección
            overlay = frame_raw.copy()
            overlay[mask_clean > 0] = [0, 255, 0]

            cv2.putText(
                overlay,
                "Enter/Q: Salir",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            horizontal = np.hstack((overlay,frame_raw))

            # Mostrar la imagen convirtiéndola correctamente a BGR para la ventana de OpenCV
            cv2.imshow(window_name, cv2.cvtColor(horizontal, cv2.COLOR_RGB2BGR))


            tecla = cv2.waitKey(10) & 0xFF
            if tecla == 13 or tecla == ord("q"):
                break

    finally:
        camara.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()