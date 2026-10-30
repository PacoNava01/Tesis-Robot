"""
Script de Seguimiento de marcadores ArUco con GUI y Gráfica PyQtGraph
Desarrollado originalmente para proyecto UnamSpace,
con la variante de control de cámara original de la Tesis de Francisco Nava.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import os

import cv2
import cv2.aruco as aruco
import numpy as np
from picamera2 import Picamera2

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

import pyqtgraph as pg

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

# --- Parámetros de Servomotores ---
PULSE_WIDTH_RANGE = (405, 2500)
START_ANGLE = 90.0
ANGLE_X_LIMIT = [45, 135]
ANGLE_Y_LIMIT = [50, 110]


def create_aruco_detector(dict_type=aruco.DICT_6X6_100):
    dictionary = aruco.getPredefinedDictionary(dict_type)
    if hasattr(aruco, "ArucoDetector"):
        parameters = aruco.DetectorParameters()
        return aruco.ArucoDetector(dictionary, parameters), True
    parameters = aruco.DetectorParameters_create()
    return (dictionary, parameters), False


def detect_aruco_markers(frame: np.ndarray, detector_info):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector, new_api = detector_info
    if new_api:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids, rejected


def estimate_marker_pose(corners, ids, marker_length, camera_matrix, dist_coeffs):
    half = marker_length / 2.0
    object_points = np.array([
        [-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]
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
    if ids is None:
        return frame
    axis_length = marker_length
    axis_points = np.array([
        [0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]
    ], dtype=np.float32)

    for pose, marker_corners, marker_id in zip(poses, corners, ids.flatten()):
        aruco.drawDetectedMarkers(frame, [marker_corners], np.array([[marker_id]]))
        img_points, _ = cv2.projectPoints(axis_points, pose.rvec, pose.tvec, camera_matrix, dist_coeffs)
        img_points = img_points.astype(int)

        origin = tuple(img_points[0].ravel())
        cv2.line(frame, origin, tuple(img_points[1].ravel()), (0, 0, 255), 3)
        cv2.line(frame, origin, tuple(img_points[2].ravel()), (0, 255, 0), 3)
        cv2.line(frame, origin, tuple(img_points[3].ravel()), (255, 0, 0), 3)

        pitch, roll, yaw = rotation_to_euler(pose.rvec)
        corner = marker_corners[0][0].astype(int)
        cv2.putText(frame, f"ID: {marker_id}", tuple(corner + [0, -50]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Dist: {pose.distance:.2f} m", tuple(corner + [0, -30]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def dibujar_centros(frame, corners, ids):
    if ids is None:
        return frame
    for marcador_id, corner in zip(ids.flatten(), corners):
        puntos = corner.reshape(4, 2)
        cx, cy = int(np.mean(puntos[:, 0])), int(np.mean(puntos[:, 1]))
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    return frame


def obtener_centro_objetivo(corners, ids, target_id):
    if ids is None:
        return None
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == target_id:
            puntos = corners[i].reshape(4, 2)
            return int(np.mean(puntos[:, 0])), int(np.mean(puntos[:, 1]))
    return None


def filtrar_error(error_actual: float, error_nuevo: float, alpha: float = 0.2) -> float:
    return error_actual * (1 - alpha) + error_nuevo * alpha


class VentanaArUcoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seguimiento ArUco - Panel de Control con Gráfica PID")
        self.resize(1100, 650)

        # Inicialización de Hardware y Visión
        self.inicializar_hardware_y_vision()
        
        # Variables de control e historiales de error
        self.angle_x, self.angle_y = START_ANGLE, START_ANGLE
        self.last_detection = None
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.error_x_filtrado = 0.0
        self.error_y_filtrado = 0.0

        self.historial_error_x = deque(maxlen=MAX_HISTORIAL)
        self.historial_error_y = deque(maxlen=MAX_HISTORIAL)

        # --- Construcción de la Interfaz ---
        central = QWidget()
        self.setCentralWidget(central)
        layout_principal = QVBoxLayout(central)

        layout_visuales = QHBoxLayout()

        # 1. Visor de video
        self.video_label = QLabel("Iniciando flujo de video...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; color: #aaa; border: 1px solid #444;")
        layout_visuales.addWidget(self.video_label)

        # 2. Widget de Gráfica con PyQtGraph
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumSize(400, 480)
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.setTitle("Gráfica de Error PID (X e Y)", color="w", size="11pt")
        self.plot_widget.setLabel("left", "Error (píxeles)", color="w")
        self.plot_widget.setLabel("bottom", "Muestras", color="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        self.curve_x = self.plot_widget.plot(pen=pg.mkPen(color=(255, 100, 0), width=2), name="Error X")
        self.curve_y = self.plot_widget.plot(pen=pg.mkPen(color=(0, 200, 0), width=2), name="Error Y")

        layout_visuales.addWidget(self.plot_widget)
        layout_principal.addLayout(layout_visuales)

        # 3. Panel de métricas de texto
        layout_metricas = QHBoxLayout()
        self.lbl_distancia = QLabel("Distancia: ---")
        self.lbl_error_x = QLabel("Error X: ---")
        self.lbl_error_y = QLabel("Error Y: ---")
        for lbl in (self.lbl_distancia, self.lbl_error_x, self.lbl_error_y):
            lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #333;")
            layout_metricas.addWidget(lbl)
        layout_principal.addLayout(layout_metricas)

        # 4. Interruptor de Modo de Prueba
        self.switch_prueba = QCheckBox("Modo prueba distancia (Servos fijos en 90°)")
        self.switch_prueba.setStyleSheet("font-size: 12px; margin-top: 5px;")
        layout_principal.addWidget(self.switch_prueba)

        # Temporizador principal para procesar frames (~30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.procesar_frame)
        self.timer.start(30)

    def inicializar_hardware_y_vision(self):
        try:
            self.camara = Camara.init_cam()
            if self.camara is None:
                raise RuntimeError("No se pudo inicializar la cámara.")
            print("Cámara inicializada correctamente.")
        except Exception as e:
            print(f"Error crítico en cámara: {e}")
            sys.exit(1)

        try:
            kit = MG_servo.init_servos(2, PULSE_WIDTH_RANGE, 180)
            self.servo_x = kit.servo[0]
            self.servo_y = kit.servo[1]
            print("2 servos inicializados correctamente.")
        except Exception as e:
            print(f"Error en servos: {e}")
            sys.exit(1)
        os.system("clear")
        try:
            self.mtx, self.dist = Apply_cam_calib.cargar_calibracion(RUTAS["Json_calib"])
            print("Calibración cargada exitosamente.")
        except Exception as e:
            print(f"Error cargando calibración: {e}")
            sys.exit(1)

        self.detector_info = create_aruco_detector(aruco.DICT_6X6_100)
        self.pid_x = PID(kP=0.007, kI=0.0001, kD=0.0030)
        self.pid_y = PID(kP=0.006, kI=0.0010, kD=0.0020)

    def procesar_frame(self):
        frame_raw = self.camara.capture_array()
        if frame_raw is None:
            return

        frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_180)
        frame = frame_raw.copy()

        corners, ids, rejected = detect_aruco_markers(frame_raw, self.detector_info)
        poses = estimate_marker_pose(corners, ids, MARKER_LENGTH, self.mtx, self.dist)

        if ids is not None:
            frame = draw_pose(frame, corners, ids, poses, MARKER_LENGTH, self.mtx, self.dist)
            frame = dibujar_centros(frame, corners, ids)

        centro = obtener_centro_objetivo(corners, ids, TARGET_MARKER_ID)
        distancia_objetivo = next((p.distance for p in poses if p.marker_id == TARGET_MARKER_ID), None)

        if distancia_objetivo is not None:
            self.lbl_distancia.setText(f"Distancia: {distancia_objetivo:.2f} m")
        else:
            self.lbl_distancia.setText("Distancia: ---")

        modo_prueba = self.switch_prueba.isChecked()

        if centro is not None:
            cx, cy = centro
            self.last_detection = time.monotonic()

            error_x = CENTER_X - cx
            error_y = CENTER_Y - cy
            self.last_error_x, self.last_error_y = error_x, error_y

            self.error_x_filtrado = filtrar_error(self.error_x_filtrado, error_x)
            self.error_y_filtrado = filtrar_error(self.error_y_filtrado, error_y)

            self.lbl_error_x.setText(f"Error X: {error_x:.1f}")
            self.lbl_error_y.setText(f"Error Y: {error_y:.1f}")

            if modo_prueba:
                self.angle_x, self.angle_y = START_ANGLE, START_ANGLE
                cv2.putText(frame, "[MODO PRUEBA] Servos Fijos", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                self.angle_x, _ = MG_servo.actualizar_eje(
                    error=self.error_x_filtrado, dead_zone=DEAD_ZONE, pid_controller=self.pid_x,
                    current_angle=self.angle_x, limits=ANGLE_X_LIMIT, servo_id=self.servo_x
                )
                self.angle_y, _ = MG_servo.actualizar_eje(
                    error=self.error_y_filtrado, dead_zone=DEAD_ZONE, pid_controller=self.pid_y,
                    current_angle=self.angle_y, limits=ANGLE_Y_LIMIT, servo_id=self.servo_y
                )
        else:
            if self.last_detection is not None and (time.monotonic() - self.last_detection) < DETECTION_TIMEOUT:
                cv2.putText(frame, "Pérdida momentánea", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "Marcador no detectado", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.historial_error_x.append(self.error_x_filtrado)
        self.historial_error_y.append(self.error_y_filtrado)

        self.curve_x.setData(list(self.historial_error_x))
        self.curve_y.setData(list(self.historial_error_y))

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        """Detiene el temporizador y asegura la liberación segura de la cámara."""
        print("Cerrando recursos de hardware y cámara...")
        try:
            self.timer.stop()  # 1. Detenemos el reloj de eventos primero
            self.camara.stop() # 2. Detenemos el stream de la cámara de forma segura
        except Exception as e:
            print(f"Nota al cerrar recursos: {e}")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = VentanaArUcoGUI()
    ventana.show()
    sys.exit(app.exec_())