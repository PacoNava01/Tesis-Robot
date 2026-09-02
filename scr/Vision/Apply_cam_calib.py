from pathlib import Path
import json
import cv2
import numpy as np
from picamera2 import Picamera2

# Rutas de archivos y configuración
JSON_CALIBRACION = Path("/home/pacon/Tesis-Robot/scr/Vision/data_calib/raspberry_pi_hq_calibration.json")
RESOLUCION = (640, 480)


def cargar_calibracion(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga la matriz de la cámara y los coeficientes de distorsión desde un JSON."""
    if not json_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de calibración en: {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    mtx = np.array(data["camera_matrix"], dtype=np.float32)
    # Usamos la llave correcta y aplanamos el arreglo a 1D
    dist = np.array(data["distortion_coefficients"], dtype=np.float32).flatten()
    
    return mtx, dist


def init_cam() -> Picamera2 | None:
    """Inicializa y configura la Picamera2."""
    try:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": RESOLUCION, "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        print("Cámara inicializada correctamente.")
        return cam
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return None


def corregir_frame(frame: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Aplica la corrección de distorsión pinhole a un frame individual."""
    # Rotar si tu montaje físico lo requiere
    frame_rotado = cv2.rotate(frame, cv2.ROTATE_180)
    
    # Aplicar corrección de distorsión
    frame_corregido = cv2.undistort(frame_rotado, mtx, dist, None, mtx)
    
    return frame_corregido


def main():
    # 1. Cargar parámetros de calibración
    try:
        mtx, dist = cargar_calibracion(JSON_CALIBRACION)
        print("Parámetros de calibración cargados con éxito.")
    except Exception as e:
        print(e)
        return

    # 2. Inicializar hardware (Cámara)
    camara = init_cam()
    if camara is None:
        return

    # 3. Configurar interfaz de visualización
    nombre_ventana = "Camara Corregida (PinHole)"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_AUTOSIZE)

    try:
        # 4. Bucle principal de procesamiento
        while True:
            frame_raw = camara.capture_array()
            if frame_raw is None:
                print("Error: No se pudo capturar el frame.")
                break

            # Aplicar bloque de corrección geométrica
            frame_corregido = corregir_frame(frame_raw, mtx, dist)

            # Convertir de RGB (Picamera2) a BGR (para que OpenCV muestre bien los colores)
            frame_bgr = cv2.cvtColor(frame_corregido, cv2.COLOR_RGB2BGR)

            # Mostrar resultado
            cv2.imshow(nombre_ventana, frame_corregido)

            # Salir con la tecla Enter (13) o 'q'
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 13 or tecla == ord("q"):
                break

    finally:
        # 5. Limpieza de recursos
        print("Liberando recursos...")
        camara.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()