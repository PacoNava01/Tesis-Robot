# Librerías generales
import csv
import json
from pathlib import Path
import time
import cv2
import numpy as np
from picamera2 import Picamera2

# Librerías para tareas de CV
from scr.Vision import Apply_cam_calib
from scr.Vision import Camara
from scr.Vision import Detection_SVM_LUT

# Librerías para tareas de hardware
from scr.Hardware import Motores_DC_correct
from scr.Hardware import MG966R_control as MG_servo
from scr.Hardware.PID_camera import PID

# --- Rutas y configuración de archivos ---
RUTAS = {
    "Lut_path": Path(
        "/home/pacon/Tesis-Robot/scr/Vision/data_calib/color_lut.npy"
    ),
    "Json_calib": Path(
        "/home/pacon/Tesis-Robot/scr/Vision/data_calib/raspberry_pi_hq_calibration.json"
    ),
}

CSV_FILE_PATH = Path(
    "/home/pacon/Tesis-Robot/docs/data/registro_errores_tilt-pan_v2.csv"
)

# --- Parámetros de la cámara y control ---
DEFAULT_K_AREA = 169889133.06
DEFAULT_B_AREA = -20.94
DEFAULT_FX = 909.897352
DEFAULT_W_REAL = 15.0  # Ancho físico real del marcador en centímetros
AREA_DETECT_TARGET = 1200

TARGET_AREA = 153000  # Área ideal que debe ocupar el objeto en píxeles
AREA_DEAD_ZONE = 0.05  # 5% de tolerancia
DEAD_ZONE = 5
DETECTION_TIMEOUT = 0.2

# --- Parámetros de servomotores y chasis ---
PINES_IZQ = (17, 27, 12)  # Forward, Backward, Enable (PWM)
PINES_DER = (23, 22, 13)  # Forward, Backward, Enable (PWM)
PIN_STBY = 24

PULSE_WIDTH_RANGE = (405, 2500)
START_ANGLE = 90.0
SERVO_DEAD_ZONE_GIRAR = 20.0  # Si pasa de 70° o 110°, el chasis gira
ANGLE_X_LIMIT = [45, 135]
ANGLE_Y_LIMIT = [50, 110]

FRAME_W, FRAME_H = 640, 480
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2


# --- Funciones auxiliares de control y filtrado ---
def calcular_velocidades(
    error_area, angle_x, start_angle, pid_linear, servo_dead_zone_girar
):
    if abs(error_area) > AREA_DEAD_ZONE:
        velocidad_lineal = pid_linear.update(error_area)
        velocidad_lineal = np.clip(velocidad_lineal, -1.0, 1.0)
    else:
        velocidad_lineal = 0.0

    if abs(angle_x - start_angle) > servo_dead_zone_girar:
        velocidad_giro = 0.4 if angle_x > start_angle else -0.4
    else:
        velocidad_giro = 0.0

    vel_izq = round(
        np.clip(velocidad_lineal - velocidad_giro, -1.0, 1.0), 2
    )
    vel_der = round(
        np.clip(velocidad_lineal + velocidad_giro, -1.0, 1.0), 2
    )

    return vel_izq, vel_der


def zona_muerta(error, base=5, factor=0.05):
    """Evita que el robot intente corregir microerrores irrelevantes."""
    return base + factor * abs(error)


def filtrar_error(error_actual, error_nuevo, alpha=0.2):
    """Filtro exponencial aplicado al error para suavizar el movimiento de servos y motores."""
    return error_actual * (1 - alpha) + error_nuevo * alpha


def suavizar_velocidad(vel_actual, vel_nueva, alpha=0.3):
    """Filtro exponencial para suavizar cambios bruscos de velocidad."""
    return vel_actual * (1 - alpha) + vel_nueva * alpha


def main():
    # Asegurar la creación del directorio de logs
    CSV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Inicializar dispositivos
    camara = Camara.init_cam()
    if camara is None:
        print("Error crítico: No se pudo inicializar la cámara.")
        return

    carro = Motores_DC_correct.Carro(PINES_IZQ, PINES_DER, PIN_STBY)
    kit = MG_servo.init_servos(2, PULSE_WIDTH_RANGE, 180)
    servo_x = kit.servo[0]  # Horizontal
    servo_y = kit.servo[1]  # Vertical

    # Cargar calibración de la cámara
    try:
        mtx, dist = Apply_cam_calib.cargar_calibracion(RUTAS["Json_calib"])
        print("Parámetros de calibración cargados exitosamente.")
    except Exception as e:
        print(f"Error cargando la calibración: {e}")
        return

    # Inicializar controladores PID
    pid_x = PID(kP=0.015, kI=0.01, kD=0.0005)
    pid_y = PID(kP=0.015, kI=0.01, kD=0.0005)
    pid_linear = PID(kP=0.6, kI=0.01, kD=0.05)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    window_name = "Deteccion"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Abrir archivo de registro CSV
    with open(
        CSV_FILE_PATH, mode="w", newline="", encoding="utf-8"
    ) as archivo_log:
        writer = csv.writer(archivo_log)
        writer.writerow([
            "timestamp",
            "error_x",
            "error_y",
            "u_x",
            "u_y",
            "error_area",
            "angle_x",
            "angle_y",
            "estado_tracking",
        ])

        vel_izq_actual, vel_der_actual = 0.0, 0.0
        error_x_filtrado, error_y_filtrado, error_area_filtrado = (
            0.0,
            0.0,
            0.0,
        )
        angle_x, angle_y = START_ANGLE, START_ANGLE
        last_detection = time.time()

        try:
            while True:
                frame_raw = camara.capture_array()
                if frame_raw is None:
                    print("Advertencia: Frame nulo recibido de la cámara.")
                    break

                estado_tracking = 0
                frame_raw = Apply_cam_calib.corregir_frame(
                    frame_raw, mtx, dist
                )
                mask, frame_bgr = Detection_SVM_LUT.Preprocess(
                    frame_raw, rotate=False
                )

                mask_clean = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, kernel
                )
                mask_clean = cv2.morphologyEx(
                    mask_clean, cv2.MORPH_CLOSE, kernel
                )
                mask_clean = cv2.GaussianBlur(mask_clean, (11, 11), 0)

                best_centroid, area, _, c = (
                    Detection_SVM_LUT.calcular_centroide_y_area(
                        mask_clean, AREA_DETECT_TARGET
                    )
                )

                if best_centroid is not None:
                    last_detection = time.time()

                overlay = frame_bgr.copy()
                mask_principal = np.zeros_like(mask_clean)

                if c is not None and area > AREA_DETECT_TARGET:
                    cv2.drawContours(
                        mask_principal, [c], -1, 255, thickness=cv2.FILLED
                    )
                    color_mask = np.zeros_like(frame_bgr)
                    color_mask[mask_principal > 0] = [100, 255, 0]
                    overlay = cv2.addWeighted(
                        frame_bgr, 1.0, color_mask, 0.35, 0
                    )

                # --- CONTROL DE MOTORES Y SERVOS ---
                if best_centroid or (
                    time.time() - last_detection < DETECTION_TIMEOUT
                ):
                    if best_centroid:
                        estado_tracking = 1
                        error_x = CENTER_X - best_centroid[0]
                        error_y = CENTER_Y - best_centroid[1]
                        error_area = (TARGET_AREA - area) / TARGET_AREA
                        error_area = np.clip(
                            error_area, -1.0, 1.0
                        )  # Corrección aplicada aquí
                    else:
                        error_x = pid_x.last_error
                        error_y = pid_y.last_error
                        error_area = getattr(pid_linear, "last_error", 0)

                    # Filtrado de errores
                    error_x_filtrado = filtrar_error(error_x_filtrado, error_x)
                    error_y_filtrado = filtrar_error(error_y_filtrado, error_y)
                    error_area_filtrado = filtrar_error(error_area_filtrado, error_area)

                    # Actualizar servos
                    angle_x, adjust_x = MG_servo.actualizar_eje(
                        error=error_x_filtrado,
                        dead_zone=DEAD_ZONE,
                        pid_controller=pid_x,
                        current_angle=angle_x,
                        limits=ANGLE_X_LIMIT,
                        servo_id=servo_x,
                    )
                    angle_y, adjust_y = MG_servo.actualizar_eje(
                        error=error_y_filtrado,
                        dead_zone=DEAD_ZONE,
                        pid_controller=pid_y,
                        current_angle=angle_y,
                        limits=ANGLE_Y_LIMIT,
                        servo_id=servo_y,
                    )

                    # Guardar log (descomentar si se requiere registrar en tiempo real)
                    # writer.writerow([time.time(), error_x, error_y, adjust_x, adjust_y, error_area, angle_x, angle_y, estado_tracking])

                    # Calcular y suavizar velocidades
                    vel_izq_nueva, vel_der_nueva = calcular_velocidades(
                        error_area_filtrado,
                        angle_x,
                        START_ANGLE,
                        pid_linear,
                        SERVO_DEAD_ZONE_GIRAR,
                    )
                    vel_izq_actual = suavizar_velocidad(
                        vel_izq_actual, vel_izq_nueva
                    )
                    vel_der_actual = suavizar_velocidad(
                        vel_der_actual, vel_der_nueva
                    )

                    # carro.mover(vel_izq_actual, vel_der_actual)

                    # Dibujar elementos visuales en pantalla
                    if c is not None:
                        cv2.drawContours(overlay, [c], -1, (255, 0, 0), 2)
                    if best_centroid:
                        cv2.circle(
                            overlay, best_centroid, 5, (0, 0, 255), -1
                        )

                    cv2.putText(overlay,"Siguiendo",(15, 50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1,)
                    cv2.putText(overlay,
                        f"Area/Target/Error: {round(area,2)}, {TARGET_AREA}, {error_area:.2f}",
                        (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1,)
                    
                    cv2.putText(
                        overlay,
                        f"Velocidad: {vel_izq_actual:.2f}, {vel_der_actual:.2f}",
                        (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),1,)

                else:
                    vel_izq_actual = suavizar_velocidad(vel_izq_actual, 0.0)
                    vel_der_actual = suavizar_velocidad(vel_der_actual, 0.0)
                    carro.mover(vel_izq_actual, vel_der_actual)

                cv2.putText(
                    overlay,
                    "Enter/Q: Salir",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (250, 0, 100),
                    1,
                )
                horizontal = np.hstack((
                    overlay,
                    cv2.cvtColor(mask_principal, cv2.COLOR_GRAY2BGR),
                ))
                cv2.imshow(window_name, horizontal)

                tecla = cv2.waitKey(10) & 0xFF
                if tecla == 13 or tecla == ord("q"):
                    break

        finally:
            camara.stop()
            cv2.destroyAllWindows()
            print(
                f"Datos estadísticos guardados exitosamente en: {CSV_FILE_PATH}"
            )


if __name__ == "__main__":
    main()