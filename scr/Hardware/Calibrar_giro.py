#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de Calibración de Factor de Giro (v, w) para Robot 4WD
Basado en Motores_DC_correct.py y control cinemático continuo.

Este script permite ajustar y probar interactivamente el factor de giro (sensibilidad
de rotación) en tiempo real para calibrar el comportamiento físico de tu robot.
"""

import sys
import time
from gpiozero import Motor, OutputDevice

class CarroCalibracion:
    # Coeficientes obtenidos de tu calibración experimental para ir en línea recta
    COEFS_DERECHO = {
        'a': 0.9356779218031275,
        'b': -1.864449111530371,
        'c': 1.1723900277729624,
        'd': -0.2070794160209165
    }

    def __init__(self, factor_giro_inicial=0.5):
        # Configuración física usando los pines de tu script Motores_DC_correct.py
        # Izquierdo: Forward=17, Backward=27, Enable=12 (PWM)
        # Derecho: Forward=23, Backward=22, Enable=13 (PWM)
        print("\n[!] Inicializando motores y puente H...")
        self.motor_izq = Motor(forward=17, backward=27, enable=12)
        self.motor_der = Motor(forward=23, backward=22, enable=13)
        self.stby = OutputDevice(24)
        self.stby.on() # Activar Standby en el puente H
        
        self.factor_giro = factor_giro_inicial
        print(f"[OK] Robot listo. Pines: Izq(17,27,12) | Der(23,22,13) | STBY=24")
        print(f"[OK] Factor de giro inicial establecido en: {self.factor_giro:.2f}")

    def compensar_derecho(self, v_deseada):
        """Aplica tu calibración polinómica al motor derecho preservando el sentido."""
        if abs(v_deseada) < 0.05:  # Zona muerta pequeña para evitar ruidos
            return 0.0
            
        signo = 1.0 if v_deseada >= 0 else -1.0
        v_abs = abs(v_deseada)
        
        # Tu polinomio de calibración
        v_comp = (self.COEFS_DERECHO['a'] * (v_abs**3) + 
                  self.COEFS_DERECHO['b'] * (v_abs**2) + 
                  self.COEFS_DERECHO['c'] * v_abs + 
                  self.COEFS_DERECHO['d'])
                  
        # Restringir la salida al rango físico del PWM [0.0, 1.0]
        v_comp = max(0.0, min(1.0, v_comp))
        return v_comp * signo

    def _set_motor(self, motor, valor_pwm):
        """Envía la señal física de dirección y PWM al motor."""
        if valor_pwm >= 0:
            motor.forward(valor_pwm)
        else:
            motor.backward(abs(valor_pwm))

    def conducir(self, v, omega):
        """
        Conducción diferencial usando cinemática diferencial con factor_giro dinámico.
        v: Velocidad lineal deseada (-1.0 a 1.0)
        omega: Velocidad angular deseada (-1.0 a 1.0, positivo gira a la izquierda)
        """
        # Calcular velocidades teóricas para cada lado usando el factor de giro actual
        v_izq = v - (omega * self.factor_giro)
        v_der = v + (omega * self.factor_giro)

        # Escalar/Limitar valores para no saturar el PWM fuera de [-1.0, 1.0]
        max_val = max(abs(v_izq), abs(v_der), 1.0)
        v_izq /= max_val
        v_der /= max_val

        # Compensar el motor derecho con tu calibración polinómica
        v_der_comp = self.compensar_derecho(v_der)

        # Aplicar el movimiento físico
        self._set_motor(self.motor_izq, v_izq)
        self._set_motor(self.motor_der, v_der_comp)

    def ejecutar_test(self, v, omega, duracion):
        """Ejecuta un movimiento controlado por un tiempo determinado y luego frena."""
        print(f"\n[>] Ejecutando test: v={v:.2f}, w={omega:.2f} con factor_giro={self.factor_giro:.2f} por {duracion:.1f} segundos...")
        self.conducir(v, omega)
        time.sleep(duracion)
        self.detener()
        print("[*] Test finalizado. Freno aplicado.")

    def detener(self):
        """Detiene inmediatamente ambos motores."""
        self.motor_izq.stop()
        self.motor_der.stop()

    def liberar_gpios(self):
        """Apaga STBY y detiene motores."""
        self.detener()
        self.stby.off()
        print("\n[!] GPIOs liberados y puente H desactivado.")

def mostrar_menu(factor_giro):
    print("\n" + "="*55)
    print(f"      CALIBRADOR DE FACTOR DE GIRO EN TIEMPO REAL")
    print("="*55)
    print(f"  --> Factor de Giro Actual: {factor_giro:.2f} <--")
    print("="*55)
    print("  [1] Modificar Factor de Giro")
    print("  [2] Test: Avance en línea recta (v=0.5, w=0.0) - 1.5s")
    print("  [3] Test: Giro sobre su propio eje IZQ (v=0.0, w=0.7) - 1.2s")
    print("  [4] Test: Giro sobre su propio eje DER (v=0.0, w=-0.7) - 1.2s")
    print("  [5] Test: Curva suave IZQUIERDA (v=0.5, w=0.4) - 1.5s")
    print("  [6] Test: Curva suave DERECHA (v=0.5, w=-0.4) - 1.5s")
    print("  [7] Control de teclado básico (W: adelante, S: atrás, A: izq, D: der, Q: stop)")
    print("  [8] Detener Motores")
    print("  [9] Salir de la calibración")
    print("="*55)

def control_teclado(robot):
    print("\n" + "*"*45)
    print("           CONTROL MANUAL DE TECLADO")
    print("*"*45)
    print(" Usa las siguientes teclas y presiona ENTER:")
    print("  [w] Adelante     [s] Atrás")
    print("  [a] Izquierda    [d] Derecha")
    print("  [q] Detener      [x] Volver al menú")
    print("*"*45)
    
    while True:
        try:
            tecla = input("Comando > ").strip().lower()
            if tecla == 'w':
                robot.conducir(v=0.5, omega=0.0)
            elif tecla == 's':
                robot.conducir(v=-0.5, omega=0.0)
            elif tecla == 'a':
                robot.conducir(v=0.2, omega=0.6)  # Curva a la izquierda
            elif tecla == 'd':
                robot.conducir(v=0.2, omega=-0.6) # Curva a la derecha
            elif tecla == 'q':
                robot.detener()
                print("[*] Robot detenido.")
            elif tecla == 'x':
                robot.detener()
                break
            else:
                print("[!] Tecla no reconocida (usa w, s, a, d, q, o x para volver)")
        except Exception as e:
            print(f"[ERROR] Ocurrió un problema en el control: {e}")
            robot.detener()
            break

def main():
    # Inicializar robot
    try:
        robot = CarroCalibracion(factor_giro_inicial=0.5)
    except ModuleNotFoundError:
        print("\n[!] ADVERTENCIA: No se encontró la librería 'gpiozero'.")
        print("    Este script debe ejecutarse directamente en tu Raspberry Pi.")
        print("    Para simular localmente, puedes instalar gpiozero o ejecutarlo en el robot.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Error al inicializar el hardware: {e}")
        sys.exit(1)

    try:
        while True:
            mostrar_menu(robot.factor_giro)
            opcion = input("Selecciona una opción [1-9]: ").strip()

            if opcion == '1':
                try:
                    nuevo_factor = float(input("\nIntroduce el nuevo factor de giro (sugerido 0.3 a 1.2): ").strip())
                    if nuevo_factor <= 0:
                        print("[!] El factor debe ser mayor que cero.")
                    else:
                        robot.factor_giro = nuevo_factor
                        print(f"[OK] Factor de giro actualizado a: {robot.factor_giro:.2f}")
                except ValueError:
                    print("[!] Por favor, introduce un número decimal válido.")

            elif opcion == '2':
                robot.ejecutar_test(v=0.5, omega=0.0, duracion=1.5)

            elif opcion == '3':
                robot.ejecutar_test(v=0.0, omega=0.7, duracion=1.2)

            elif opcion == '4':
                robot.ejecutar_test(v=0.0, omega=-0.7, duracion=1.2)

            elif opcion == '5':
                robot.ejecutar_test(v=0.5, omega=0.4, duracion=1.5)

            elif opcion == '6':
                robot.ejecutar_test(v=0.5, omega=-0.4, duracion=1.5)

            elif opcion == '7':
                control_teclado(robot)

            elif opcion == '8':
                robot.detener()
                print("\n[*] Motores detenidos.")

            elif opcion == '9':
                break

            else:
                print("[!] Opción inválida. Elige un número del 1 al 9.")

    except KeyboardInterrupt:
        print("\n\n[!] Interrupción detectada. Saliendo de forma segura...")
    finally:
        # Asegurar frenado seguro y liberación de pines
        robot.liberar_gpios()

if __name__ == "__main__":
    main()
