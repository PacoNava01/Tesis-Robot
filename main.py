#Librerias generales
import cv2
from pathlib import Path
import json
import numpy as np
from picamera2 import Picamera2
import time

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



#Parametros del carrito
pines_izq = (17, 27, 12)  # Forward, Backward, Enable (PWM)
pines_der = (22, 23, 13)  # Forward, Backward, Enable (PWM)
pin_stby = 24

#--- Parmetros camara ---
DEFAULT_K_AREA = 169889133.06
DEFAULT_B_AREA = -20.94
DEFAULT_FX = 909.897352
DEFAULT_W_REAL = 15.0  # Ancho físico real del marcador en centímetros (15 cm)

#--- Umbrales --- en HSV
umbrales = {"rojo_low1":[100,100,100],
            "rojo_low2":[100,100,100],
            "rojo_up1":[100,100,100],
            "rojo_up2":[100,100,100],
            "azul":[120,121,121]}


#---PID parmetros---
pid_x = PID(kP=0.01, kI=0.02, kD=0.0005) #servo x
pid_y = PID(kP=0.01, kI=0.02, kD=0.0005) #servo y
pid_linear = PID(kP=0.05, kI=0.001, kD=0.01)#velocidad lineal usando el PID camara
TARGET_AREA = 25000  # Área ideal que debe ocupar el objeto en píxeles (ajustar según la cámara)
AREA_DEAD_ZONE = 1000
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


if camara is None:
    print("Un problema ocurrió con la cámara")

window_name = "Deteccion"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

try:
    mtx,dist = Apply_cam_calib.cargar_calibracion(rutas["Json_calib"])
    print("Parametro de calibracion cargados exitosamente")
    
    while True:
        frame_raw = camara.capture_array()
        if frame_raw is None:
            break

        # Aplicar calibración de cámara
        frame_raw = Apply_cam_calib.corregir_frame(frame_raw, mtx, dist)
        
        mask, frame_bgr = Detection_SVM_LUT.Preprocess(frame_raw, rotate=False)

        # Aplicamos limpieza morfológica
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.GaussianBlur(mask_clean, (11, 11), 0)
        
        best_centroid, area, last_detection, c = Detection_SVM_LUT.calcular_centroide_y_area(mask_clean, 1200)     

        if best_centroid is not None:
            last_detection = time.time()
                 
        # Superponer la detección
        overlay = frame_bgr.copy()
        overlay[mask_clean > 0] = [0, 255, 0]
        
        # Inicializar variables de movimiento del chasis por defecto en 0
        velocidad_lineal = 0.0
        velocidad_giro = 0.0

        # Dibujar contorno y centroide si se detecta algo o pasa el tiempo para considerar una captura
        if best_centroid or (time.time() - last_detection < detection_timeout):
            if best_centroid:
                error_x = CENTER_X - best_centroid[0]
                error_y = CENTER_Y - best_centroid[1]
                error_area = TARGET_AREA - area  # Positivo si está lejos, Negativo si está muy cerca
                
                if abs(error_area) > AREA_DEAD_ZONE:
                    velocidad_lineal = pid_linear.update(error_area)
                    # Limitar velocidad lineal a los rangos permitidos
                    velocidad_lineal = max(-1.0, min(1.0, velocidad_lineal))
                else:
                    velocidad_lineal = 0.0
            else:
                # Si se perdió momentáneamente pero estamos en el timeout de inercia
                error_x = pid_x.last_error
                error_y = pid_y.last_error
                error_area = getattr(pid_linear, 'last_error', 0)
                velocidad_lineal = 0.0
            
            # --- 1. Actualizar eje X (Pan) y eje Y (Tilt) ---
            angle_x = MG_servo.actualizar_eje(error=error_x,
                                              dead_zone=dead_zone,
                                              pid_controller=pid_x,
                                              current_angle=angle_x,
                                              limits=angle_x_limit,
                                              servo_id=servo_x)

            angle_y = MG_servo.actualizar_eje(error=error_y,
                                              dead_zone=dead_zone,
                                              pid_controller=pid_y,
                                              current_angle=angle_y,
                                              limits=angle_y_limit,
                                              servo_id=servo_y)

            # --- 2. Control Angular del Chasis (Efecto "Cuello Humano") ---
            if abs(angle_x - start_angle) > SERVO_DEAD_ZONE_GIRAR:
                if angle_x > start_angle:
                    velocidad_giro = 0.4   # Girar hacia la derecha
                else:
                    velocidad_giro = -0.4  # Girar hacia la izquierda

            # --- 3. Combinar y Enviar al Carro ---
            vel_izq = velocidad_lineal - velocidad_giro
            vel_der = velocidad_lineal + velocidad_giro
            
            # Limitar a los rangos máximos del puente H (-100 a 100)
            vel_izq = max(-1.0, min(1.0, vel_izq))
            vel_der = max(-1.0, min(1.0, vel_der))
            
            # Enviar comandos a los motores DC
            # carro.mover(vel_izq, vel_der)

        else:
            # Si pasó el tiempo de tolerancia sin detectar nada, detener el chasis por seguridad
            carro.detener()
            pass

            cv2.drawContours(overlay, c, -1, (255, 0, 0), 2)  # Contorno azul
            
            # Dibujar centroide
            cv2.circle(overlay, best_centroid, 5, (0, 0, 255), -1)  # Círculo rojo relleno
            cv2.putText(overlay, "siguiendo", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"Area: {area}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        

        cv2.putText(
            overlay,
            "Enter/Q: Salir",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (250, 0, 100),
            1,
            cv2.LINE_AA,
        )
        
        #horizontal = np.hstack((overlay, frame_bgr))
        horizontal = np.hstack((overlay, cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)))
        cv2.imshow(window_name, horizontal)
        
        tecla = cv2.waitKey(10) & 0xFF
        if tecla == 13 or tecla == ord("q"):
            break

finally:
    camara.stop()
    cv2.destroyAllWindows()