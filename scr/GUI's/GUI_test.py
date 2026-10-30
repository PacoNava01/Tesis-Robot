import sys
import os
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

class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simulación GUI - Seguimiento ArUco")
        self.resize(700, 600)

        # Widget central y layout principal
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 1. Visor de video simulado (simulando el frame de la cámara)
        self.video_label = QLabel("Simulación de Video (Cámara)")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color: #1e1e1e; color: #aaaaaa; border: 2px dashed #555; font-size: 14px;"
        )
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # 2. Panel de métricas (Distancia y Errores)
        layout_metricas = QHBoxLayout()
        self.lbl_distancia = QLabel("Distancia: ---")
        self.lbl_error_x = QLabel("Error X: ---")
        self.lbl_error_y = QLabel("Error Y: ---")

        # Darles un poco de estilo limpio a las etiquetas
        for lbl in (self.lbl_distancia, self.lbl_error_x, self.lbl_error_y):
            lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #333;")
            layout_metricas.addWidget(lbl)

        layout.addLayout(layout_metricas)

        # 3. Interruptor de pruebas
        self.switch_prueba = QCheckBox("Modo prueba distancia (Servos fijos en 90°)")
        self.switch_prueba.setStyleSheet("font-size: 12px; margin-top: 5px;")
        self.switch_prueba.stateChanged.connect(self.cambiar_modo_prueba)
        layout.addWidget(self.switch_prueba)

        # 4. Timer para simular la actualización en tiempo real (~30 FPS / cada 33 ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self.simular_actualizacion)
        self.timer.start(33)

    def cambiar_modo_prueba(self, estado):
        """Muestra en consola cuando activas o desactivas el checkbox."""
        if self.switch_prueba.isChecked():
            print("[SIMULACIÓN] Modo prueba activado: Los servos se fijarán a 90°.")
        else:
            print("[SIMULACIÓN] Modo prueba desactivado: Seguimiento PID activo.")

    def simular_actualizacion(self):
        """Simula la llegada de datos de distancia y errores de prueba."""
        modo_prueba_activo = self.switch_prueba.isChecked()

        # Cambiamos valores de prueba de manera dinámica
        self.lbl_distancia.setText("Distancia: 0.75 m")
        
        if modo_prueba_activo:
            self.lbl_error_x.setText("Error X: 0.0 (Fijo)")
            self.lbl_error_y.setText("Error Y: 0.0 (Fijo)")
        else:
            self.lbl_error_x.setText("Error X: +15.2 (PID)")
            self.lbl_error_y.setText("Error Y: -8.4 (PID)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    ventana = VentanaPrincipal()
    ventana.show()
    
    # Usar sys.exit con app.exec_() asegura la limpieza correcta de hilos
    sys.exit(app.exec_())