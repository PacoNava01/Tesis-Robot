"""
Script de Seguimiento de marcadores ArUco
Desarrollado originalmente para proyecto UnamSpace,
con la variante de control de cámara original de la Tesis de Francisco Nava.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import cv2.aruco as aruco
import numpy as np
from picamera2 import Picamera2

# Librerías para tareas de CV
from scr.Vision import Apply_cam_calib, Camara

# Librerías para tareas de hardware
from scr.Hardware import Motores_DC_correct
from scr.Hardware import MG966R_control as MG_servo
from scr.Hardware.PID_camera import PID


@dataclass
class MarkerPose:
    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    distance: float


# --- Configuración y Rutas ---
RUTAS = {
    "Json_calib": Path(
        "/home/pacon/Tesis-Robot/scr/Vision/data_calib/raspberry_pi_hq_calibration.json"
    )
}

# --- Parámetros de la Cámara y Control ---
FRAME_W, FRAME_H = 640, 480
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2

MARKER_LENGTH = 0.10  # Ancho físico real del marcador en metros
TARGET_MARKER_ID = 2  # ID del marcador a seguir
DETECTION_TIMEOUT = 0.2
DEAD_ZONE = 10
MAX_HISTORIAL = 150

# --- Bandera de Modo de Prueba ---
MODO_PRUEBA_DISTANCIA = True  # True: Servos fijos en 90° solo midiendo | False: Seguimiento PID activo

# --- Parámetros de Servomotores ---
PULSE_WIDTH_RANGE = (405, 2500)
START_ANGLE = 90.0
ANGLE_X_LIMIT = [45, 135]
ANGLE_Y_LIMIT = [50, 110]


def generar_grafica_error(hist_x: deque, hist_y: deque, ancho: int = 600, alto: int = 200) -> np.ndarray:
    """Crea una imagen con la gráfica en tiempo real de los errores X e Y."""
    canvas = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # Fondo gris oscuro

    centro_y = alto // 2
    cv2.line(canvas, (0, centro_y), (ancho, centro_y), (80, 80, 80), 1)

    cv2.putText(canvas, "Error = 0", (10, centro_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    cv2.putText(canvas, "X: Azul | Y: Verde", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if len(hist_x) < 2:
        return canvas

    paso_x = ancho / MAX_HISTORIAL
    escala_y = (alto / 2) / 320.0

    puntos_x = [
        (int(i * paso_x), max(0, min(alto - 1, int(centro_y - (err * escala_y)))))
        for i, err in enumerate(hist_x)
    ]
    for i in range(len(puntos_x) - 1):
        cv2.line(canvas, puntos_x[i], puntos_x[i + 1], (255, 100, 0), 2)

    puntos_y = [
        (int(i * paso_x), max(0, min(alto - 1, int(centro_y - (err * escala_y)))))
        for i, err in enumerate(hist_y)
    ]
    for i in range(len(puntos_y) - 1):
        cv2.line(canvas, puntos_y[i], puntos_y[i + 1], (0, 200, 0), 2)

    return canvas


def create_aruco_detector(dict_type=aruco.DICT_6X6_100):
    """Crea el detector ArUco compatible con distintas versiones de OpenCV."""
    dictionary = aruco.getPredefinedDictionary(dict_type)

    if hasattr(aruco, "ArucoDetector"):
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector, True

    parameters = aruco.DetectorParameters_create()
    return (dictionary, parameters), False


def detect_aruco_markers(frame: np.ndarray, detector_info):
    """Detecta marcadores ArUco en un frame convirtiéndolo a escala de grises."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector, new_api = detector_info

    if new_api:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    return corners, ids, rejected


def estimate_marker_pose(corners, ids, marker_length, camera_matrix, dist_coeffs):
    """Estima la pose y distancia de los marcadores ArUco detectados."""
    half = marker_length / 2.0
    object_points = np.array([
        [-half, half, 0],
        [half, half, 0],
        [half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

    poses = []
    if ids is None:
        return poses

    for i, marker_corners in enumerate(corners):
        marker_id = int(ids[i][0])
        image_points = marker_corners[0].astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            continue
        distance = float(np.linalg.norm(tvec))
        poses.append(MarkerPose(marker_id=marker_id, rvec=rvec, tvec=tvec, distance=distance))

    return poses


def rotation_to_euler(rvec: np.ndarray) -> np.ndarray:
    """Convierte el vector de rotación (rvec) a ángulos Euler en grados."""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        roll = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        roll = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0

    return np.degrees([pitch, roll, yaw])


def draw_pose(frame, corners, ids, poses, marker_length, camera_matrix, dist_coeffs):
    """Dibuja los ejes 3D, el ID, la distancia y la orientación sobre los marcadores."""
    if ids is None:
        return frame

    axis_length = marker_length
    axis_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ], dtype=np.float32)

    for pose, marker_corners, marker_id in zip(poses, corners, ids.flatten()):
        aruco.drawDetectedMarkers(frame, [marker_corners], np.array([[marker_id]]))

        img_points, _ = cv2.projectPoints(axis_points, pose.rvec, pose.tvec, camera_matrix, dist_coeffs)
        img_points = img_points.astype(int)

        origin = tuple(img_points[0].ravel())
        x_axis = tuple(img_points[1].ravel())
        y_axis = tuple(img_points[2].ravel())
        z_axis = tuple(img_points[3].ravel())

        cv2.line(frame, origin, x_axis, (0, 0, 255), 3)
        cv2.line(frame, origin, y_axis, (0, 255, 0), 3)
        cv2.line(frame, origin, z_axis, (255, 0, 0), 3)

        pitch, roll, yaw = rotation_to_euler(pose.rvec)
        corner = marker_corners[0][0].astype(int)

        cv2.putText(frame, f"ID: {marker_id}", tuple(corner + [0, -50]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Dist: {pose.distance:.2f} m", tuple(corner + [0, -30]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"P:{pitch:.0f} R:{roll:.0f} Y:{yaw:.0f}", tuple(corner + [0, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return frame


def dibujar_centros(frame, corners, ids):
    """Dibuja un círculo y coordenadas en el centro de cada marcador."""
    if ids is None:
        return frame

    for marcador_id, corner in zip(ids.flatten(), corners):
        puntos = corner.reshape(4, 2)
        cx, cy = int(np.mean(puntos[:, 0])), int(np.mean(puntos[:, 1]))
        
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"ID:{marcador_id} Centro:({cx},{cy})", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    return frame


def obtener_centro_objetivo(corners, ids, target_id):
    """Retorna las coordenadas (cx, cy) del ID objetivo, o None si no se detecta."""
    if ids is None:
        return None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == target_id:
            puntos = corners[i].reshape(4, 2)
            cx = int(np.mean(puntos[:, 0]))
            cy = int(np.mean(puntos[:, 1]))
            return cx, cy

    return None


def filtrar_error(error_actual: float, error_nuevo: float, alpha: float = 0.2) -> float:
    """Aplica un filtro exponencial al error para suavizar el movimiento de los servos."""
    return error_actual * (1 - alpha) + error_nuevo * alpha


def main():
    """Función principal para inicializar hardware y ejecutar el bucle de control."""
    try:
        camara = Camara.init_cam()
        if camara is None:
            raise RuntimeError("No se pudo inicializar la cámara. Revisa las conexiones.")
    except Exception as e:
        print(f"Error crítico en cámara: {e}")
        return

    try:
        kit = MG_servo.init_servos(2, PULSE_WIDTH_RANGE, 180)
        servo_x = kit.servo[0]
        servo_y = kit.servo[1]
    except Exception as e:
        print(f"Error al inicializar los servos: {e}")
        camara.stop()
        return

    try:
        mtx, dist = Apply_cam_calib.cargar_calibracion(RUTAS["Json_calib"])
        print("Parámetros de calibración de cámara cargados exitosamente.")
    except Exception as e:
        print(f"Error cargando la calibración: {e}")
        camara.stop()
        return

    detector_info = create_aruco_detector(aruco.DICT_6X6_100)

    pid_x = PID(kP=0.007, kI=0.0001, kD=0.0030)
    pid_y = PID(kP=0.006, kI=0.0010, kD=0.0020)

    angle_x, angle_y = START_ANGLE, START_ANGLE
    
    historial_error_x = deque(maxlen=MAX_HISTORIAL)
    historial_error_y = deque(maxlen=MAX_HISTORIAL)

    last_detection = None
    last_error_x = 0.0
    last_error_y = 0.0
    error_x_filtrado = 0.0
    error_y_filtrado = 0.0

    if MODO_PRUEBA_DISTANCIA:
        print(">>> MODO DE PRUEBA DE DISTANCIA ACTIVADO: Servos fijos en 90° <<<")
    else:
        print(">>> MODO DE SEGUIMIENTO PID ACTIVADO <<<")

    try:
        while True:
            frame_raw = camara.capture_array()
            if frame_raw is None:
                print("Advertencia: Frame nulo recibido.")
                break

            frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_180)
            frame = frame_raw.copy()

            corners, ids, rejected = detect_aruco_markers(frame_raw, detector_info)
            poses = estimate_marker_pose(corners, ids, MARKER_LENGTH, mtx, dist)

            if ids is not None:
                frame = draw_pose(frame, corners, ids, poses, MARKER_LENGTH, mtx, dist)
                frame = dibujar_centros(frame, corners, ids)

            centro = obtener_centro_objetivo(corners, ids, TARGET_MARKER_ID)

            # Obtener distancia del marcador objetivo
            distancia_objetivo = next((p.distance for p in poses if p.marker_id == TARGET_MARKER_ID), None)
            if distancia_objetivo is not None:
                print(f"Distancia al objetivo (ID {TARGET_MARKER_ID}): {distancia_objetivo:.2f} m")

            if centro is not None:
                cx, cy = centro
                last_detection = time.monotonic()

                error_x = CENTER_X - cx
                error_y = CENTER_Y - cy

                last_error_x = error_x
                last_error_y = error_y

                error_x_filtrado = filtrar_error(error_x_filtrado, error_x)
                error_y_filtrado = filtrar_error(error_y_filtrado, error_y)

                # --- CONTROL CONDICIONADO POR LA BANDERA ---
                if MODO_PRUEBA_DISTANCIA:
                    # En modo prueba dejamos los servos fijos en 90° (ángulo de inicio)
                    # Y mostramos un aviso en pantalla
                    angle_x, angle_y = START_ANGLE, START_ANGLE
                    cv2.putText(frame, "[MODO PRUEBA] Servos Fijos (90 deg)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    # Funcionamiento normal: Actualización mediante PID
                    angle_x, _ = MG_servo.actualizar_eje(
                        error=error_x_filtrado, dead_zone=DEAD_ZONE, pid_controller=pid_x,
                        current_angle=angle_x, limits=ANGLE_X_LIMIT, servo_id=servo_x
                    )
                    angle_y, _ = MG_servo.actualizar_eje(
                        error=error_y_filtrado, dead_zone=DEAD_ZONE, pid_controller=pid_y,
                        current_angle=angle_y, limits=ANGLE_Y_LIMIT, servo_id=servo_y
                    )

                    cv2.circle(frame, (CENTER_X, CENTER_Y), 5, (255, 0, 0), -1)
                    cv2.line(frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 255, 255), 2)
                    cv2.putText(frame, f"ID:{TARGET_MARKER_ID} Centro:({cx},{cy})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.putText(frame, f"Error X:{error_x:.1f} Y:{error_y:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            else:
                estado_tracking = 0
                if last_detection is not None and (time.monotonic() - last_detection) < DETECTION_TIMEOUT:
                    error_x = last_error_x
                    error_y = last_error_y
                    cv2.putText(frame, "Pérdida momentánea del marcador", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    cv2.putText(frame, "Marcador no detectado", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            historial_error_x.append(error_x_filtrado)
            historial_error_y.append(error_y_filtrado)

            img_grafica = generar_grafica_error(historial_error_x, historial_error_y)
            cv2.imshow("Grafica Error PID", img_grafica)
            cv2.imshow("Seguimiento ArUco", frame)

            tecla = cv2.waitKey(10) & 0xFF
            if tecla in (13, ord("q")):
                break

    except Exception as e:
        print(f"Error inesperado en ejecución: {e}")

    finally:
        print("Cerrando cámara y liberando recursos de vídeo...")
        camara.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    