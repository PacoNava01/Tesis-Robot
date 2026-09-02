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

#Rutas de archivos 
rutas = {
    "Lut_path": Path("/home/pacon/Tesis-Robot/scr/Vision/data_calib/color_lut.npy"),
    "Json_calib": Path("/home/pacon/Tesis-Robot/scr/Vision/data_calib/raspberry_pi_hq_calibration.json")
}

#Parametros del carrito
pines_izq = (17, 27, 12)  # Forward, Backward, Enable (PWM)
pines_der = (22, 23, 13)  # Forward, Backward, Enable (PWM)
pin_stby = 24

# Definir el kernel correctamente como matriz de OpenCV
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#Constantes de Resolucion y centro
FRAME_W, FRAME_H = 640, 480
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2

#Inicializar camara y robot
camara = Camara.init_cam()
carro = Motores_DC_correct.Carro(pines_izq, pines_der, pin_stby)

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

		#Aplicar calibración de camara
        frame_raw = Apply_cam_calib.corregir_frame(frame_raw,mtx,dist)
		
        mask, frame_bgr = Detection_SVM_LUT.Preprocess(frame_raw,rotate=False)

        # Aplicamos limpieza morfologica
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        best_centroid, area = Detection_SVM_LUT.calcular_centroide_y_area(mask)		

        # Superponer la deteccion
        overlay = frame_bgr.copy()
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
        
        horizontal = np.hstack((overlay, frame_bgr))
        cv2.imshow(window_name, horizontal)
        
        tecla = cv2.waitKey(10) & 0xFF
        if tecla == 13 or tecla == ord("q"):
            break

finally:
    camara.stop()
    cv2.destroyAllWindows()