import cv2
import numpy as np
import glob
import pandas as pd
import os

'''
Este script se encarga únicamente de abrir imágenes, permitir la selección de ROI y guardar el dataset en CSV.
'''

class PixelDatasetGenerator:
    def __init__(self, images_path="/home/pacon/Tesis-Robot/scr/Vision/data_calib/Red_color_photos/*.jpg", color_space="HSV"):
        self.images_path = images_path
        self.color_space = color_space.upper()
        self.positives = []
        self.negatives = []

    def collect_from_rois(self):
        images = glob.glob(self.images_path)
        if len(images) == 0:
            print("No se encontraron imágenes en la ruta.")
            return False

        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            if img is None:
                continue

            display_img = img.copy()
            cv2.putText(display_img, f"Img {idx+1}/{len(images)}: Selecciona ROI",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            roi = cv2.selectROI(f"Imagen {idx+1}", display_img, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(f"Imagen {idx+1}")

            x, y, rw, rh = roi
            if rw < 2 or rh < 2:
                continue

            if self.color_space == "HSV":
                converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            else:
                converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            crop = converted_img[y:y+rh, x:x+rw]
            self.positives.extend(crop.reshape(-1,3))

            mask_negative = np.ones(img.shape[:2], dtype=np.uint8) * 255
            pad = 10
            mask_negative[max(0,y-pad):min(img.shape[0], y+rh+pad),
                          max(0,x-pad):min(img.shape[1], x+rw+pad)] = 0

            neg_idx = np.where(mask_negative > 0)
            all_negatives = converted_img[neg_idx[0], neg_idx[1]]
            num_neg_to_sample = min(len(all_negatives), len(crop.reshape(-1,3))*2)
            if num_neg_to_sample > 0:
                sampled_negatives = all_negatives[np.random.choice(len(all_negatives), num_neg_to_sample, replace=False)]
                self.negatives.extend(sampled_negatives)

        return len(self.positives) > 0 and len(self.negatives) > 0

    def save_dataset(self, filename="pixel_dataset.csv"):
        X_pos = np.array(self.positives, dtype=np.float32)
        X_neg = np.array(self.negatives, dtype=np.float32)
        y_pos = np.ones((X_pos.shape[0],), dtype=np.int32)
        y_neg = np.zeros((X_neg.shape[0],), dtype=np.int32)

        X = np.vstack((X_pos, X_neg))
        y = np.concatenate((y_pos, y_neg))

        if self.color_space == "HSV":
            df = pd.DataFrame(X, columns=["H","S","V"])

        else:
            df = pd.DataFrame(X, columns=["B","G","R"])

        
        df["Label"] = y
        df.to_csv(filename, index=False)
        print(f"Dataset guardado en {filename}")

if __name__ == "__main__":
    generator = PixelDatasetGenerator(images_path="/home/pacon/Tesis-Robot/scr/Vision/data_calib/Red_color_photos/*.jpg", color_space="HSV")
    if generator.collect_from_rois():
        generator.save_dataset("pixel_dataset.csv")
    else:
        print("No se recolectaron suficientes muestras.")
