import cv2
import numpy as np
import joblib
import argparse
import os
import json


'''
Script enfocado en la deteccion de objetos de color rojo a partir de la segmentacion por threshold de mascaras HSV, ademas de otras funciones de segmentaciomn y distancia


'''
# -------------------------------
# Funciones de segmentación y distancia
# -------------------------------

def get_red_mask(hsv_img):
    """
    Genera una máscara binaria para el color rojo en HSV.
    Combina dos rangos de matiz (0-10 y 170-180).
    """
    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_img, lower_red_1, upper_red_1)

    lower_red_2 = np.array([170, 100, 100])
    upper_red_2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_img, lower_red_2, upper_red_2)

    return cv2.bitwise_or(mask1, mask2)

def apply_morphological_operations(mask):
    """
    Limpia la máscara con apertura y cierre morfológico.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

def estimate_distance_area_model(area, k=169889133.06, b=-20.94):
    if (area - b) <= 0:
        return float('nan')
    return np.sqrt(k / (area - b))

def estimate_distance_linear_model(width_px, f_x=909.897352, W_real=15.0):
    if width_px <= 0:
        return float('nan')
    return (f_x * W_real) / width_px

# -------------------------------
# Clase Detector con ML
# -------------------------------

class ObjectDetector:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def extract_features(self, contour, h_channel, s_channel, v_channel):
        area = cv2.contourArea(contour)
        if area == 0:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

        mask_roi = np.zeros_like(h_channel)
        cv2.drawContours(mask_roi, [contour], -1, 255, -1)
        h_mean = cv2.mean(h_channel, mask=mask_roi)[0]
        s_mean = cv2.mean(s_channel, mask=mask_roi)[0]
        v_mean = cv2.mean(v_channel, mask=mask_roi)[0]

        return np.array([area, aspect_ratio, circularity, solidity,
                         h_mean, s_mean, v_mean], dtype=np.float32)

    def process_frame(self, frame, f_x=909.897352, W_real=15.0, k_area=169889133.06, b_area=-20.94):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = get_red_mask(hsv)
        mask_cleaned = apply_morphological_operations(mask)

        h_ch, s_ch, v_ch = cv2.split(hsv)
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates_features, candidates_contours = [], []
        for c in contours:
            if cv2.contourArea(c) < 400:
                continue
            feat = self.extract_features(c, h_ch, s_ch, v_ch)
            if feat is not None:
                candidates_features.append(feat)
                candidates_contours.append(c)

        output_mask = np.zeros_like(mask_cleaned)
        if candidates_features:
            features_scaled = self.scaler.transform(candidates_features)
            predictions = self.model.predict(features_scaled)

            for i, is_target in enumerate(predictions):
                if is_target == 1:
                    c = candidates_contours[i]
                    cv2.drawContours(output_mask, [c], -1, 255, -1)

                    # Estimación de distancia
                    area = cv2.contourArea(c)
                    x, y, w, h = cv2.boundingRect(c)
                    dist_area = estimate_distance_area_model(area, k=k_area, b=b_area)
                    dist_linear = estimate_distance_linear_model(w, f_x=f_x, W_real=W_real)

                    # Visualización
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
                    cv2.putText(frame, f"Dist Area: {dist_area:.1f} cm", (x, y-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    cv2.putText(frame, f"Dist Linear: {dist_linear:.1f} cm", (x, y+h+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

        return frame, output_mask

# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="DetectoV2 - Detección ML + Distancia")
    parser.add_argument("--source", type=str, default="0", help="Video o índice de cámara")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo entrenado (joblib)")
    parser.add_argument("--scaler", type=str, required=True, help="Ruta al scaler (joblib)")
    args = parser.parse_args()

    detector = ObjectDetector(args.model, args.scaler)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la fuente {source}")
        return

    print("Iniciando DetectoV2. Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame, mask = detector.process_frame(frame)
        cv2.imshow("DetectoV2 - Salida", output_frame)
        cv2.imshow("DetectoV2 - Mascara", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
