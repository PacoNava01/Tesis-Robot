import cv2
import numpy as np
import joblib

class ObjectDetector:
    """
    Detector de objetos utilizando extracción de características
    geométricas/de color y un modelo preentrenado (ej. SVM, Random Forest).
    """
    def __init__(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def extract_features(self, contour, h_channel, s_channel, v_channel) -> np.ndarray | None:
        """
        Extrae características geométricas y de color de un contorno dado.
        """
        area = cv2.contourArea(contour)
        if area == 0:
            return None

        # --- 1. Geometría ---
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        perimeter = cv2.arcLength(contour, True)
        # Ecuación de circularidad: 4 * pi * Area / Perímetro^2
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

        # --- 2. Color (Extracción segura mediante máscara) ---
        # Es mucho más preciso usar la máscara exacta del contorno 
        # en lugar del bounding box (ROI cuadrada).
        mask_roi = np.zeros_like(h_channel)
        cv2.drawContours(mask_roi, [contour], -1, 255, -1)

        # cv2.mean devuelve una tupla (val1, val2, val3, val4). 
        # Extraemos el primer elemento [0] ya que enviamos canales individuales.
        h_mean = cv2.mean(h_channel, mask=mask_roi)[0]
        s_mean = cv2.mean(s_channel, mask=mask_roi)[0]
        v_mean = cv2.mean(v_channel, mask=mask_roi)[0]

        return np.array([
            area, 
            aspect_ratio, 
            circularity, 
            solidity, 
            h_mean, 
            s_mean, 
            v_mean
        ], dtype=np.float32)

    def process_frame(self, frame_hsv, hsv_mask) -> np.ndarray:
        """
        Procesa el frame HSV y su máscara para predecir y filtrar los objetos objetivo.
        Devuelve una máscara final solo con los objetos detectados por el modelo.
        """
        # Separar canales HSV para la extracción de características de color
        h_ch, s_ch, v_ch = cv2.split(frame_hsv)
        
        # Encontrar contornos en la máscara inicial
        contornos, _ = cv2.findContours(
            hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        output_mask = np.zeros_like(hsv_mask)
        candidates_features = []
        candidates_contours = []

        # Recopilar características de los contornos viables
        for c in contornos:
            if cv2.contourArea(c) < 400:  # Filtro inicial de ruido por tamaño
                continue
            
            feat = self.extract_features(c, h_ch, s_ch, v_ch)
            if feat is not None:
                candidates_features.append(feat)
                candidates_contours.append(c)
        
        # --- Predicción por Batch ---
        if candidates_features:
            # Escalar características
            features_scaled = self.scaler.transform(candidates_features)
            
            # Predecir sobre todos los candidatos
            predictions = self.model.predict(features_scaled)

            # Dibujar solo los contornos que el modelo clasifica como '1' (target)
            for i, is_target in enumerate(predictions):
                if is_target == 1:
                    cv2.drawContours(output_mask, [candidates_contours[i]], -1, 255, -1)
        
        return output_mask


def obtener_mask(frame_hsv, low_hsv, up_hsv) -> np.ndarray:
    """
    Genera una máscara limpia basada en un rango HSV y operaciones morfológicas.
    """
    mask = cv2.inRange(frame_hsv, low_hsv, up_hsv)
    
    # Operaciones morfológicas para limpiar ruido
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
    
    # Suavizado de bordes
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask