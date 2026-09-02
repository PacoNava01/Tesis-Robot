"""
Script de Calibración de Factor de Giro (v, w) para Robot 4WD
Basado en el script funcional y control cinemático continuo con protección de giro.
"""
import math
import sys
import time
from gpiozero import Motor, OutputDevice

class CarroCalibracion:
    COEFS_DERECHO = {
        'a': 0.9356779218031275,
        'b': -1.864449111530371,
        'c': 1.1723900277729624,
        'd': -0.2070794160209165
    }

    def __init__(self, left_pins: tuple, right_pins: tuple, stby_pin: int, invertir_stby: bool = False, factor_giro_inicial=0.5):
        print("\n[!] Inicializando motores y puente H...")
        
        self.stby = OutputDevice(stby_pin, active_high=not invertir_stby)
        self.stby.on()  # Activar Standby en el puente H

        self.motor_izq = Motor(forward=left_pins[0], backward=left_pins[1], enable=left_pins[2])
        self.motor_der = Motor(forward=right_pins[0], backward=right_pins[1], enable=right_pins[2])
        
        self.factor_giro = factor_giro_inicial
        print(f"[OK] Robot listo. Pines: Izq{left_pins} | Der{right_pins} | STBY={stby_pin}")
        print(f"[OK] Factor de giro inicial establecido en: {self.factor_giro:.2f}")

    def compensar_derecho(self, v_deseada):
        """Aplica la compensación polinómica segura (idéntica a la del script funcional)."""
        magnitud = abs(v_deseada)
        
        if magnitud < 0.4:
            return v_deseada
            
        a = self.COEFS_DERECHO['a']
        b = self.COEFS_DERECHO['b']
        c = self.COEFS_DERECHO['c']
        d = self.COEFS_DERECHO['d']
        
        x = max(0.4, min(1.0, magnitud))
        error = a * (x**3) + b * (x**2) + c * x + d
        factor = 1 - error
        
        v_comp = magnitud * factor - 0.2
        return math.copysign(max(0.0, min(1.0, abs(v_comp))), v_deseada)

    def _set_motor(self, motor, valor_pwm):
        """Envía la señal física de dirección y PWM al motor."""
        if valor_pwm >= 0:
            motor.forward(valor_pwm)
        else:
            motor.backward(abs(valor_pwm))

    def conducir(self, v, omega):
        """
        Conducción diferencial usando cinemática diferencial con factor_giro dinámico
        y protección para evitar aplicar compensación asimétrica durante giros en seco (spin).
        """
        v_izq = v - (omega * self.factor_giro)
        v_der = v + (omega * self.factor_giro)

        max_val = max(abs(v_izq), abs(v_der), 1.0)
        v_izq /= max_val
        v_der /= max_val

        # FILTRO DE PROTECCIÓN: Si v_izq y v_der tienen signos opuestos, 
        # el robot está girando sobre su propio eje. Omitimos la compensación 
        # asimétrica para evitar desbalancear o hundir el motor derecho en su zona muerta.
        if v_izq * v_der < 0:
            v_der_comp = v_der
        else:
            v_der_comp = self.compensar_derecho(v_der)

        self._set_motor(self.motor_izq, v_izq)
        self._set_motor(self.motor_der, v_der_comp)

    def ejecutar_test(self, v, omega, duracion):
        print(f"\n[>] Ejecutando test: v={v:.2f}, w={omega:.2f} con factor_giro={self.factor_giro:.2f} por {duracion:.1f} segundos...")
        self.conducir(v, omega)
        time.sleep(duracion)
        self.detener()
        print("[*] Test finalizado. Freno aplicado.")

    def detener(self):
        self.motor_izq.stop()
        self.motor_der.stop()

    def liberar_gpios(self):
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
                robot.conducir(v=0.2, omega=0.0)
            elif tecla == 's':
                robot.conducir(v=-0.3, omega=0.0)
            elif tecla == 'a':
                robot.conducir(v=0.0, omega=0.6)  # Giro limpio sobre su propio eje
            elif tecla == 'd':
                robot.conducir(v=0.0, omega=-0.6) # Giro limpio sobre su propio eje
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
    try:
        pines_izq = (17, 27, 12)  # Forward, Backward, Enable (PWM)
        pines_der = (22, 23, 13)  # Forward, Backward, Enable (PWM)
        pin_stby = 24
        
        robot = CarroCalibracion(pines_izq, pines_der, pin_stby, factor_giro_inicial=0.5)

    except ModuleNotFoundError:
        print("\n[!] ADVERTENCIA: No se encontró la librería 'gpiozero'.")
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
        robot.liberar_gpios()

if __name__ == "__main__":
    main()