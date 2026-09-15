#Librerias generales
import cv2
from pathlib import Path
import json
import numpy as np
from picamera2 import Picamera2
import time
import csv

#Librerias para tareas de CV
from scr.Vision import Detection_SVM_LUT
from scr.Vision import Camara
from scr.Vision import Apply_cam_calib

#Librerias para tareas de hardware
from scr.Hardware.PID_camera import PID
from scr.Hardware import Motores_DC_correct
from scr.Hardware import MG966R_control as MG_servo
#Rutas de archivos 
rutas = {
    "Lut_path": Path("/home/pacon/Tesis-Robot/scr/Vision/data_calib/color_lut.npy"),
    "Json_calib": Path("/home/pacon/Tesis-Robot/scr/Vision/data_calib/raspberry_pi_hq_calibration.json")
}

# --- Configuración de registro estadístico ---
csv_file_path = Path("/home/pacon/Tesis-Robot/docs/data/registro_errores_tilt-pan_v2.csv")
csv_file_path.parent.mkdir(parents=True, exist_ok=True) # Asegura que la carpeta exista

archivo_log = open(csv_file_path, mode="w", newline="", encoding="utf-8")
writer = csv.writer(archivo_log)
# Escribir la cabecera del archivo CSV
writer.writerow(["timestamp", "error_x", "error_y","u_x","u_y","error_area", "angle_x", "angle_y","estado_tracking"])#, "vel_lineal", "vel_giro"])

#Parametros del carrito
pines_izq = (17, 27, 12)  # Forward, Backward, Enable (PWM)
pines_der = (22, 23, 13)  # Forward, Backward, Enable (PWM)
pin_stby = 24

#--- Parmetros camara ---
DEFAULT_K_AREA = 169889133.06
DEFAULT_B_AREA = -20.94
DEFAULT_FX = 909.897352
DEFAULT_W_REAL = 15.0  # Ancho físico real del marcador en centímetros (15 cm)
area_detect_target = 1200

#--- Umbrales --- en HSV
umbrales = {"rojo_low1":[100,100,100],
            "rojo_low2":[100,100,100],
            "rojo_up1":[100,100,100],
            "rojo_up2":[100,100,100],
            "azul":[120,121,121]}


#---PID parmetros---
pid_x = PID(kP=0.01, kI=0.02, kD=0.0005) #servo x
pid_y = PID(kP=0.01, kI=0.02, kD=0.0005) #servo y
pid_linear = PID(kP=0.8, kI=0.005, kD=0.1)#velocidad lineal usando el PID camara
TARGET_AREA = 153000  # Área ideal que debe ocupar el objeto en píxeles (ajustar según la cámara)
AREA_DEAD_ZONE = 0.05 #½ de tolerancia
dead_zone = 5
detection_timeout = 0.2
last_detection = time.time()

#--- Parametros servomotores ---
PULSE_WIDTH_RANGE = (405, 2500)
start_angle = 90.0
SERVO_DEAD_ZONE_GIRAR = 20.0  # Si pasa de 70° o 110°, el chasis gira
angle_x,angle_y = 90.0,90.0
angle_x_limit = [45,135]
angle_y_limit = [50,110]

# Definir el kernel correctamente como matriz de OpenCV
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#Constantes de Resolucion y centro
FRAME_W, FRAME_H = 640, 480
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2

#Inicializar camara , robot y servos
camara = Camara.init_cam()
carro = Motores_DC_correct.Carro(pines_izq, pines_der, pin_stby)
kit =  MG_servo.init_servos(2,PULSE_WIDTH_RANGE,180)
servo_x = kit.servo[0] #Horizontal
servo_y = kit.servo[1] #Vertical

# --- Funciones auxiliares ---
def calcular_velocidades(error_area, angle_x, start_angle, pid_linear, SERVO_DEAD_ZONE_GIRAR):
    # Control lineal
    if abs(error_area) > AREA_DEAD_ZONE:
        velocidad_lineal = pid_linear.update(error_area)
        velocidad_lineal = np.clip(velocidad_lineal, -1.0, 1.0)
    else:
        velocidad_lineal = 0.0

    # Control angular
    if abs(angle_x - start_angle) > SERVO_DEAD_ZONE_GIRAR:
        velocidad_giro = 0.4 if angle_x > start_angle else -0.4
    else:
        velocidad_giro = 0.0

    # Combinación
    vel_izq = round(np.clip(velocidad_lineal - velocidad_giro, -1.0, 1.0), 2)
    vel_der = round(np.clip(velocidad_lineal + velocidad_giro, -1.0, 1.0), 2)

    return vel_izq, vel_der


def suavizar_velocidad(vel_actual, vel_nueva, alpha=0.3):
    """Filtro exponencial para suavizar cambios bruscos"""
    return vel_actual * (1 - alpha) + vel_nueva * alpha

if camara is None:
    print("Un problema ocurrió con la cámara")

window_name = "Deteccion"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

try:
    mtx,dist = Apply_cam_calib.cargar_calibracion(rutas["Json_calib"])
    print("Parametro de calibracion cargados exitosamente")
    
# --- BLOQUE PRINCIPAL ---
vel_izq_actual, vel_der_actual = 0.0, 0.0  # Inicialización de velocidades

while True:
    frame_raw = camara.capture_array()
    if frame_raw is None:
        break

    estado_tracking = 0
    frame_raw = Apply_cam_calib.corregir_frame(frame_raw, mtx, dist)
    mask, frame_bgr = Detection_SVM_LUT.Preprocess(frame_raw, rotate=False)

    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.GaussianBlur(mask_clean, (11, 11), 0)

    best_centroid, area, last_detection, c = Detection_SVM_LUT.calcular_centroide_y_area(
        mask_clean, area_detect_target
    )

    if best_centroid is not None:
        last_detection = time.time()

    overlay = frame_bgr.copy()
    mask_principal = np.zeros_like(mask_clean)

    if c is not None and area > area_detect_target:
        cv2.drawContours(mask_principal, [c], -1, 255, thickness=cv2.FILLED)
        color_mask = np.zeros_like(frame_bgr)
        color_mask[mask_principal > 0] = [100, 255, 0]
        overlay = cv2.addWeighted(frame_bgr, 1.0, color_mask, 0.35, 0)

    # --- CONTROL DE MOTORES ---
    if best_centroid or (time.time() - last_detection < detection_timeout):
        if best_centroid:
            estado_tracking = 1
            error_x = CENTER_X - best_centroid[0]
            error_y = CENTER_Y - best_centroid[1]
            error_area = (TARGET_AREA - area) / TARGET_AREA
        else:
            # Fallback: usar último error registrado
            error_x = pid_x.last_error
            error_y = pid_y.last_error
            error_area = getattr(pid_linear, "last_error", 0)

        # Actualizar servos
        angle_x, adjust_x = MG_servo.actualizar_eje(
            error=error_x, dead_zone=dead_zone,
            pid_controller=pid_x, current_angle=angle_x,
            limits=angle_x_limit, servo_id=servo_x
        )
        angle_y, adjust_y = MG_servo.actualizar_eje(
            error=error_y, dead_zone=dead_zone,
            pid_controller=pid_y, current_angle=angle_y,
            limits=angle_y_limit, servo_id=servo_y
        )

        # Guardar log
        writer.writerow([
            time.time(), error_x, error_y,
            adjust_x, adjust_y, error_area,
            angle_x, angle_y, estado_tracking
        ])

        # Calcular velocidades nuevas
        vel_izq_nueva, vel_der_nueva = calcular_velocidades(error_area, angle_x, start_angle, pid_linear, SERVO_DEAD_ZONE_GIRAR)

        # Suavizar transición
        vel_izq_actual = suavizar_velocidad(vel_izq_actual, vel_izq_nueva)
        vel_der_actual = suavizar_velocidad(vel_der_actual, vel_der_nueva)

        carro.mover(vel_izq_actual, vel_der_actual)

        # Dibujos
        if c is not None:
            cv2.drawContours(overlay, [c], -1, (255, 0, 0), 2)
        if best_centroid:
            cv2.circle(overlay, best_centroid, 5, (0, 0, 255), -1)

        cv2.putText(overlay, "siguiendo", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(overlay, f"Area-Error_area: {area,TARGET_AREA,error_area}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(overlay, f"Velocidad: {vel_izq_actual:.2f},{vel_der_actual:.2f}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    else:
        # Fallback seguro: detener progresivamente
        vel_izq_actual = suavizar_velocidad(vel_izq_actual, 0.0)
        vel_der_actual = suavizar_velocidad(vel_der_actual, 0.0)
        carro.mover(vel_izq_actual, vel_der_actual)

    cv2.putText(overlay, "Enter/Q: Salir", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 100), 1)
    horizontal = np.hstack((overlay, cv2.cvtColor(mask_principal, cv2.COLOR_GRAY2BGR)))
    cv2.imshow(window_name, horizontal)

    tecla = cv2.waitKey(10) & 0xFF
    if tecla == 13 or tecla == ord("q"):
        break


finally:
    archivo_log.close() # Cierra el archivo CSV de forma segura
    camara.stop()
    cv2.destroyAllWindows()
    print(f"Datos estadísticos guardados exitosamente en: {csv_file_path}")
