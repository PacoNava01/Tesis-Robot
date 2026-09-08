import cv2
import numpy as np
import joblib
import argparse
import os
import json
import time

# =====================================================================
# CONSTANTES DE CALIBRACIÓN Y ESTIMACIÓN (Valores por defecto del usuario)
# =====================================================================
DEFAULT_K_AREA = 169889133.06
DEFAULT_B_AREA = -20.94
DEFAULT_FX = 909.897352
DEFAULT_W_REAL = 15.0  # Ancho físico real del marcador en centímetros (15 cm)

# =====================================================================
# FUNCIONES DE PROCESAMIENTO GEOMÉTRICO Y ESTIMACIÓN DE DISTANCIA
# =====================================================================
def get_optimal_undistort_maps(camera_matrix, dist_coeffs, image_size):
    """
    Precomputa los mapas de rectificación para desdistorsionar imágenes de forma óptima,
    evitando recalcularlos en cada fotograma para ahorrar CPU en la Raspberry Pi.
    """
    h, w = image_size[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )
    return mapx, mapy, new_camera_matrix

def apply_morphological_operations(mask):
    """
    Aplica filtros morfológicos de apertura (para eliminar puntos de ruido aislados)
    y cierre (para rellenar huecos dentro del marcador), terminando con un desenfoque
    Gaussiano para suavizar los bordes.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

def estimate_distance_area_model(area, k=DEFAULT_K_AREA, b=DEFAULT_B_AREA):
    """
    Modelo Experimental Cuadrático Inverso:
    Estima la distancia (D) basada en el área
    segmentada del marcador en píxeles.
    Fórmula: D = sqrt(k / (Area - b))
    """
    if (area - b) <= 0:
        return float('nan')
    return np.sqrt(k / (area - b))

def estimate_distance_linear_model(width_px, f_x=DEFAULT_FX, W_real=DEFAULT_W_REAL):
    """
    Modelo Lineal Pinhole:
    Estima la distancia (D) basada en el ancho del cuadro delimitador en píxeles.
    Fórmula: D = (f_x * W_real) / width_px
    """
    if width_px <= 0:
        return float('nan')
    return (f_x * W_real) / width_px

# =====================================================================
# CLASE DETECTOR DE OBJETOS CON MACHINE LEARNING (CON INFERENCIA HIBRIDA)
# =====================================================================
class ObjectDetector:
    def __init__(self, model_path=None, scaler_path=None, lut_path=None):
        """
        Constructor híbrido:
        - Si se especifica lut_path, se cargará la Tabla de Búsqueda 3D (.npy) para inferencia ultra-rápida.
        - De lo contrario, se cargarán el clasificador SVM RBF (.joblib) y su normalizador.
        """
        self.use_lut = False
        self.lut = None
        self.model = None
        self.scaler = None

        if lut_path and os.path.exists(lut_path):
            try:
                self.lut = np.load(lut_path)
                self.use_lut = True
                print(f"[ML INFO] Cargada Tabla de Búsqueda 3D (LUT) exitosamente desde: {lut_path}")
            except Exception as e:
                print(f"[ML WARNING] Error al cargar la LUT {lut_path}: {e}. Intentando modo clasificador tradicional.")

        if not self.use_lut:
            if model_path and os.path.exists(model_path) and scaler_path and os.path.exists(scaler_path):
                try:
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    print(f"[ML INFO] Clasificador SVM y Normalizador cargados desde: {model_path} y {scaler_path}")
                except Exception as e:
                    print(f"[ML ERROR] No se pudieron cargar los modelos .joblib: {e}")
                    raise e
            else:
                raise ValueError("Se requiere un archivo LUT (.npy) o archivos SVM + Scaler (.joblib) válidos.")

    def get_red_mask(self, hsv_img):
        """
        Genera la máscara binaria aplicando la segmentación por Machine Learning:
        - Si usa la LUT: realiza un mapeo directo de memoria en NumPy (microsegundos).
        - Si usa la SVM: escala y predice píxel por píxel (muy lento en Raspberry Pi).
        """
        h, w, c = hsv_img.shape
        
        if self.use_lut:
            # Indexación matricial directa en NumPy utilizando los valores HSV como índices de la LUT
            # H se indexa en el primer eje, S en el segundo, V en el tercero.
            mask = self.lut[hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]]
            # Garantizar que retorne una máscara binaria estándar de OpenCV (CV_8UC1)
            return (mask * 255).astype(np.uint8)
        else:
            # Inferencia clásica (Lenta, ideal solo para imágenes estáticas en CPU o depuración)
            # Reestructurar la imagen como matriz de muestras (N_píxeles, 3)
            pixels = hsv_img.reshape(-1, 3)
            # Normalizar
            pixels_scaled = self.scaler.transform(pixels)
            # Predecir
            predictions = self.model.predict(pixels_scaled)
            # Reconstruir la geometría original de la imagen
            mask = predictions.reshape(h, w).astype(np.uint8)
            return mask * 255

# =====================================================================
# FLUJO PRINCIPAL DE EJECUCIÓN (REAL-TIME DETECTOR)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="DetectoV2 - Inferencia Real-Time ML + Calibración")
    parser.add_argument("--source", type=str, default="0", help="Ruta de archivo de video o índice de cámara (ej: 0)")
    parser.add_argument("--lut", type=str, default="color_lut.npy", help="Ruta al archivo de la LUT (.npy)")
    parser.add_argument("--model", type=str, default="modelv2.joblib", help="Ruta opcional al clasificador SVM (.joblib)")
    parser.add_argument("--scaler", type=str, default="scalerv2.joblib", help="Ruta opcional al normalizador (.joblib)")
    parser.add_argument("--calibration", type=str, default="raspberry_pi_hq_calibration.json", help="Ruta al archivo JSON de calibración")
    args = parser.parse_args()

    # 1. Cargar archivo de calibración si existe
    cam_matrix = None
    dist_coeffs = None
    f_x = DEFAULT_FX
    
    if os.path.exists(args.calibration):
        try:
            with open(args.calibration, 'r') as f:
                cal_data = json.load(f)
            cam_matrix = np.array(cal_data["camera_matrix"])
            dist_coeffs = np.array(cal_data["distortion_coefficients"])
            f_x = cam_matrix[0, 0] # Focal horizontal real calibrada
            print(f"[CALIB] Calibración cargada con éxito. Focal calibrada f_x: {f_x:.2f} px")
        except Exception as e:
            print(f"[CALIB WARNING] No se pudo parsear el archivo de calibración {args.calibration}: {e}")
            print("Se utilizará el modelo de proyección pinhole con parámetros por defecto.")
    else:
        print(f"[CALIB INFO] No se encontró el archivo '{args.calibration}'. Se asume cámara ideal (sin desdistorsión).")

    # 2. Inicializar el Detector de Objetos con Machine Learning
    try:
        if os.path.exists(args.lut):
            detector = ObjectDetector(lut_path=args.lut)
        else:
            detector = ObjectDetector(model_path=args.model, scaler_path=args.scaler)
    except Exception as e:
        print(f"[FATAL ERROR] Imposible inicializar el segmentador por Machine Learning: {e}")
        return

    # 3. Inicializar captura de video
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente de video: {source}")
        return

    print("\n=== Arrancando Bucle de Detección Real-Time ===")
    print("Presiona 'q' para salir del programa.")
    print("---------------------------------------------")

    map_x, map_y = None, None
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video finalizado o fotograma vacío.")
            break

        # Inicializar mapas de rectificación con el tamaño del primer fotograma
        if cam_matrix is not None and dist_coeffs is not None and map_x is None:
            h, w = frame.shape[:2]
            map_x, map_y, _ = get_optimal_undistort_maps(cam_matrix, dist_coeffs, (h, w))
            print(f"[INFO] Resolucion detectada: {w}x{h}. Mapas de desdistorsión precomputados.")

        # Desdistorsionar el fotograma usando los mapas optimizados
        if map_x is not None and map_y is not None:
            frame_corrected = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        else:
            frame_corrected = frame.copy()

        # Procesamiento espacial
        hsv = cv2.cvtColor(frame_corrected, cv2.COLOR_BGR2HSV)
        
        # Inferencia por Machine Learning para segmentar el color
        t_start = time.time()
        mask_raw = detector.get_red_mask(hsv)
        t_infer = (time.time() - t_start) * 1000  # Latencia en milisegundos

        # Limpieza morfológica
        mask_clean = apply_morphological_operations(mask_raw)

        # Buscar contornos de los objetos segmentados
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        target_found = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Descartar falsos positivos pequeños de fondo
                continue

            # Obtener caja delimitadora
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Calcular centroide geométrico
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = x + w_box // 2, y + h_box // 2

            target_found = True

            # Estimar distancia utilizando ambos modelos
            dist_area = estimate_distance_area_model(area)
            dist_linear = estimate_distance_linear_model(w_box, f_x=f_x)

            # Dibujar resultados sobre la pantalla
            cv2.rectangle(frame_corrected, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)  # Caja en Azul
            cv2.circle(frame_corrected, (cX, cY), 5, (0, 0, 255), -1)  # Centroide en Rojo
            
            # Etiquetas de texto informativas
            cv2.putText(frame_corrected, f"Area: {int(area)} px2", (x, y - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame_corrected, f"Dist (Area): {dist_area:.1f} cm", (x, y - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame_corrected, f"Dist (Pinhole): {dist_linear:.1f} cm", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            break # Procesar únicamente el marcador más grande por fotograma

        # Calcular FPS globales
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        # Superponer estadísticas de rendimiento de hardware
        mode_text = "SVM LUT 3D (Ultra-rapido)" if detector.use_lut else "SVM RBF (Lento)"
        cv2.putText(frame_corrected, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_corrected, f"ML latency: {t_infer:.2f} ms", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_corrected, f"Modo: {mode_text}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Mostrar video corregido y máscara segmentada
        cv2.imshow("DetectoV2 - Corregida y Segmentada", frame_corrected)
        cv2.imshow("Mascara ML limpia", mask_clean)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nPrograma finalizado exitosamente.")

if __name__ == "__main__":
    main()
