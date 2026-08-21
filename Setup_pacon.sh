#!/bin/bash
set -e  # Detiene la ejecución si ocurre cualquier error

echo " Actualizando sistema..."
sudo apt update && sudo apt upgrade -y

echo " Instalando dependencias del sistema y bindings..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    libcamera-apps \
    python3-picamera2 \
    python3-opencv \
    python3-gpiozero \
    python3-lgpio \
    python3-numpy 

echo " Creando entorno virtual .pacon..."
if [ -d ".pacon" ]; then
    echo " .pacon ya existe, omitiendo creación"
else
    python3 -m venv .pacon --system-site-packages
fi

echo " Activando entorno..."
source .pacon/bin/activate

echo " Actualizando pip en el entorno..."
pip install --upgrade pip setuptools wheel

echo " Instalando paquetes de Python (ML + Adafruit)..."
pip install \
    scikit-learn \
    joblib \
    pandas \
    Adafruit-Blinka \
    adafruit-circuitpython-servokit \
    adafruit-circuitpython-pca9685

echo " Verificando instalaciones..."
python3 -c "import cv2; print('OpenCV OK')"
python3 -c "from gpiozero import LED; print('GPIOZERO OK')"
python3 -c "import sklearn; print('Scikit-learn OK')"
python3 -c "import board; print('Blinka OK')"

echo " Setup completo listo."
echo " Para trabajar en este entorno, ejecútalo en tu terminal con: source .pacon/bin/activate"
