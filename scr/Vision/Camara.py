from pathlib import Path
import time
import cv2
from picamera2 import Picamera2
import json
import numpy as np
import os

"""
Captura fotos automáticamente cada X segundos 
y las almacena en un directorio específico.
"""

# Configuración y constantes
INTERVALO_SEGUNDOS = 2.5  # Tiempo entre cada captura automática
RUTA_DESTINO = Path(
    "/home/pacon/Tesis-Robot/scr/Vision/data_calib/Red_color_photos"
)


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


def load_calibration(calibration_path):
    """
    Carga los parámetros de calibración intrínsecos de la cámara desde un archivo JSON.
    """
    if not os.path.exists(calibration_path):
        print(f"Advertencia: No se encontró el archivo de calibración '{calibration_path}'.")
        print("El script funcionará pero no corregirá la distorsión geométrica de la lente.")
        return None, None
    
    try:
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['distortion_coefficients'])
        print(f"Calibración cargada con éxito desde: {calibration_path}")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error al cargar el archivo de calibración: {e}")
        return None, None

def estimate_distance_area_model(area, k, b):
    """
    Modelo Experimental Cuadrático Inverso:
    Estima la distancia (D) basada en el área
    segmentada del marcador en píxeles.
    Fórmula: D = sqrt(k / (Area - b))
    """
    if (area - b) <= 0:
        return float('nan')
    return np.sqrt(k / (area - b))

def estimate_distance_linear_model(width_px, f_x, W_real):
    """
    Modelo Lineal Pinhole:
    Estima la distancia (D) basada en
    el ancho del cuadro
    delimitador en píxeles.
    Fórmula: D = (f_x * W_real) / width_px
    """
    if width_px <= 0:
        return float('nan')
    return (f_x * W_real) / width_px

def main():
    # Asegurar que la ruta de destino exista
    RUTA_DESTINO.mkdir(parents=True, exist_ok=True)

    camara = init_cam()
    if camara is None:
        return

    nombre_ventana = "Captura Automatica"
    cv2.namedWindow(nombre_ventana)

    contador = len(list(RUTA_DESTINO.glob("calib_*.jpg")))
    
    # Inicializar el tiempo de la última captura
    tiempo_ultima_captura = time.time()

    print(f"Iniciando capturas automáticas cada {INTERVALO_SEGUNDOS} segundos...")

    try:
        while True:
            frame_raw = camara.capture_array()
            if frame_raw is None:
                print("Error: No se pudo capturar el frame.")
                break

            # Rotar frame según necesidad física de la cámara
            frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_180)

            # Crear una copia para mostrar en pantalla con la interfaz gráfica
            frame_display = frame_raw.copy()

            # Calcular el tiempo restante para la próxima foto
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - tiempo_ultima_captura
            tiempo_restante = max(0, int(INTERVALO_SEGUNDOS - tiempo_transcurrido) + 1)

            # Texto guía y cuenta regresiva en pantalla
            cv2.putText(
                frame_display,
                f"Siguiente foto en: {tiempo_restante}s | Enter/Q: Salir",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            # Lógica para tomar la foto cuando se cumple el intervalo
            if tiempo_transcurrido >= INTERVALO_SEGUNDOS:
                nombre_archivo = RUTA_DESTINO / f"calib_{contador}.jpg"
                cv2.imwrite(str(nombre_archivo), frame_raw)

                print(f"Imagen guardada exitosamente: {nombre_archivo}")
                contador += 1
                
                # Reiniciar el temporizador
                tiempo_ultima_captura = time.time()

            # Mostrar el frame procesado
            cv2.imshow(nombre_ventana, frame_display)

            # Salir con la tecla Enter (código 13) o 'q'
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 13 or tecla == ord("q"):
                break

    finally:
        print("Liberando recursos...")
        camara.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()