from pathlib import Path
import time
import cv2
from picamera2 import Picamera2

"""
Captura una foto después de un tiempo de espera
tras hacer click izquierdo en la ventana.
"""

# Configuración y constantes
TIEMPO_ESPERA = 3
RUTA_DESTINO = Path(
    "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/src/Vision/ArUcocalib"
)

# Variables globales para el estado del mouse
temporizador_activo = False
tiempo_click = 0.0


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


def click_mouse(event, x, y, flags, param):
    """Callback para manejar el evento del mouse."""
    global temporizador_activo, tiempo_click

    if event == cv2.EVENT_LBUTTONDOWN and not temporizador_activo:
        tiempo_click = time.time()
        temporizador_activo = True
        print("¡Temporizador iniciado! Mantente quieto...")


def main():
    global temporizador_activo

    # Asegurar que la ruta de destino exista
    RUTA_DESTINO.mkdir(parents=True, exist_ok=True)

    camara = init_cam()
    if camara is None:
        return

    nombre_ventana = "Frame capturado"
    cv2.namedWindow(nombre_ventana)
    cv2.setMouseCallback(nombre_ventana, click_mouse)

    contador = len(list(RUTA_DESTINO.glob("calib_*.jpg")))

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

            # Texto guía fijo en pantalla
            cv2.putText(
                frame_display,
                "Click izq: Tomar foto | Enter: Salir",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            # Lógica del temporizador
            if temporizador_activo:
                tiempo_transcurrido = time.time() - tiempo_click
                tiempo_restante = (
                    int(TIEMPO_ESPERA - tiempo_transcurrido) + 1
                )

                if tiempo_restante > 0:
                    # Mostrar cuenta regresiva en el display
                    cv2.putText(
                        frame_display,
                        f"Foto en {tiempo_restante}",
                        (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )
                else:
                    # Guardamos 'frame_raw' limpio (SIN los textos de la cuenta regresiva)
                    nombre_archivo = RUTA_DESTINO / f"calib_{contador}.jpg"
                    cv2.imwrite(str(nombre_archivo), frame_raw)

                    print(f"Imagen guardada exitosamente: {nombre_archivo}")
                    contador += 1
                    temporizador_activo = False

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