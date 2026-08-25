from pathlib import Path
import json
import time
import cv2
import numpy as np

# --- Importación de tus módulos locales personalizados ---
from scr.Vision.Detector import ObjectDetector, obtener_mask
from scr.Vision.Camara import init_cam
from scr.Vision.Apply_cam_calib import cargar_calibracion,corregir_frame
from scr.Hardware.MG966R_control import init_servos, mover_servo
from scr.Hardware.Motores_DC_correct import Carro



# --- Clase PID robusta ---
class PID:
    def __init__(self, kP, kI, kD):
        self.kP, self.kI, self.kD = kP, kI, kD
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            return 0

        # Proporcional
        P = self.kP * error

        # Integral (anti-windup)
        self.integral += error * dt
        self.integral = max(-10, min(10, self.integral))
        I = self.kI * self.integral

        # Derivativo
        D = self.kD * (error - self.last_error) / dt

        self.last_error = error
        self.last_time = now
        return P + I + D


def main():
    # --- Configuración de rutas y archivos ---
    BASE_DIR = Path("/home/pacon/Tesis-Robot")
    JSON_CALIBRACION = BASE_DIR / "/Vision/data_calib/raspberry_pi_hq_calibration.json"
    MODEL_PATH = BASE_DIR / "data/model.pkl"
    SCALER_PATH = BASE_DIR / "data/scaler.pkl"

    # 1. Cargar calibración de la cámara Pinhole
    try:
        mtx, dist = cargar_calibracion(JSON_CALIBRACION)
        print("Calibración de cámara cargada con éxito.")
    except Exception as e:
        print(f"Aviso de calibración: {e}. Se continuará sin corrección óptica.")
        mtx, dist = None, None

    # --- Configuración de Hardware e IA ---
    pines_izq = (17, 27, 12)
    pines_der = (23, 22, 13)
    pin_stby = 24
    
    # Uso seguro del carro mediante Context Manager (si lo implementaste previamente)
    carrito = Carro(pines_izq, pines_der, pin_stby)
    detector = ObjectDetector(str(MODEL_PATH), str(SCALER_PATH))
    cam = init_cam()

    if cam is None:
        print("Error crítico: No se pudo inicializar la cámara.")
        return

    # Servo (Pan-Tilt)
    servo_x, servo_y = init_servos()

    # Constantes de resolución y centro
    FRAME_W, FRAME_H = 640, 480
    CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2

    # Variables de estado del servo
    start_angle = 90.0
    angle_x, angle_y = 90.0, 90.0
    dead_zone = 5
    last_detection_time = time.time()
    detection_timeout = 0.2
    angle_x_limit = [45, 135]

    # --- Instanciar PIDs ---
    pid_x = PID(kP=0.01, kI=0.02, kD=0.0005)
    pid_y = PID(kP=0.01, kI=0.02, kD=0.0005)

    # --- Rangos HSV (Rojo) ---
    low_red1 = np.array([0, 115, 20])
    up_red1 = np.array([10, 255, 255])
    low_red2 = np.array([170, 100, 35])
    up_red2 = np.array([180, 255, 255])

    print("Iniciando seguimiento en tiempo real...")
    mover_servo(servo_x, int(start_angle))
    mover_servo(servo_y, int(start_angle))
    
    cv2.namedWindow("Control y Detección", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frame_raw = cam.capture_array()
            if frame_raw is None:
                print("Error: Frame vacío capturado.")
                break

            # 2. Rotación física y aplicación de corrección Pinhole
            frame_rotado = cv2.rotate(frame_raw, cv2.ROTATE_180)
            
            if mtx is not None and dist is not None:
                frame = cv2.undistort(frame_rotado, mtx, dist, None, mtx)
            else:
                frame = frame_rotado

            # --- Procesamiento de Visión ---
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = obtener_mask(hsv_frame, low_red1, up_red1)
            mask2 = obtener_mask(hsv_frame, low_red2, up_red2)
            final_mask = detector.process_frame(hsv_frame, cv2.add(mask1, mask2))

            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_centroid, area = None, 0

            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > 800:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        best_centroid = (cx, cy)
                        last_detection_time = time.time()

            # --- Lógica de Control (Servos + Chasis) ---
            if best_centroid or (time.time() - last_detection_time < detection_timeout):
                if best_centroid:
                    error_x = CENTER_X - best_centroid[0]
                    error_y = CENTER_Y - best_centroid[1]
                else:
                    error_x = pid_x.last_error
                    error_y = pid_y.last_error

                # Control Eje X (Pan)
                if abs(error_x) > dead_zone:
                    adjustment_x = pid_x.update(error_x)
                    angle_x += adjustment_x
                    angle_x = max(10, min(170, angle_x))
                    mover_servo(servo_x, int(angle_x))

                # Control Eje Y (Tilt)
                if abs(error_y) > dead_zone:
                    adjustment_y = pid_y.update(error_y)
                    angle_y += adjustment_y
                    angle_y = max(60, min(130, angle_y))
                    mover_servo(servo_y, int(angle_y))

                # Control del Chasis basado en la posición del servo horizontal
                if angle_x < angle_x_limit[0]:
                    carrito.accion("izquierda")
                elif angle_x > angle_x_limit[1]:
                    carrito.accion("derecha")
                else:
                    if area < 2000:
                        carrito.accion("avanzar", velocidad=0.4)
                    elif area > 5000:
                        carrito.accion("retroceder", velocidad=0.3)
                    else:
                        carrito.detener()
            else:
                # Si no hay objeto y pasó el timeout, detener motores
                carrito.detener()

            # --- Visualización en Pantalla ---
            display_frame = frame.copy()
            if best_centroid:
                cv2.circle(display_frame, best_centroid, 10, (0, 255, 0), 2)
                cv2.putText(display_frame, "Siguiendo objeto", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Area: {int(area)}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            ml_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((display_frame, ml_mask_bgr))
            cv2.imshow("Control y Detección", combined)

            # Salir con la tecla Enter (13)
            if cv2.waitKey(1) & 0xFF == 13:
                break

    finally:
        # --- Cierre y Limpieza de Recursos ---
        print("\nDeteniendo sistema y liberando recursos...")
        carrito.detener()
        if hasattr(carrito, 'apagar_driver'):
            carrito.apagar_driver()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()