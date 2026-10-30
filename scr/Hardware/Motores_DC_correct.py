from gpiozero import Robot, Motor, OutputDevice
import math
import time

class Carro:
    """
    Controlador avanzado para un carro diferencial con compensación 
    polinomica de velocidad para el motor derecho y gestión de driver (STBY).
    """
    COEFS = {
        'a': 0.9356779218031275,
        'b': -1.864449111530371,
        'c': 1.1723900277729624,
        'd': -0.2070794160209165
    }

    def __init__(self, left_pins: tuple, right_pins: tuple, stby_pin: int, invertir_stby: bool = False):
        # Configuración del pin de standby del driver (ej. TB6612FNG o L298N)
        self.stby = OutputDevice(stby_pin, active_high=not invertir_stby)
        self.activar_driver()

        # Configuración de motores usando gpiozero
        motor_izq = Motor(forward=left_pins[0], backward=left_pins[1], enable=left_pins[2])
        motor_der = Motor(forward=right_pins[0], backward=right_pins[1], enable=right_pins[2])

        self.robot = Robot(left=motor_izq, right=motor_der)

    def __enter__(self):
        """Permite usar la clase con la sentencia 'with'."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Asegura apagar el driver de forma segura al salir del contexto."""
        self.apagar_driver()

    def _clamp(self, v: float, minimo: float = -1.0, maximo: float = 1.0) -> float:
        """Limita los valores de PWM dentro del rango físico permitido."""
        return max(minimo, min(maximo, v))

    def _compensacion(self, pwm: float) -> float:
        """Calcula el factor de corrección basado en el polinomio experimental."""
        a, b, c, d = (self.COEFS[k] for k in ('a', 'b', 'c', 'd'))
        x = max(0.4, min(1.0, pwm))
        error = a * x**3 + b * x**2 + c * x + d
        return 1 - error

    def _compensar_derecho(self, v: float) -> float:
        """Aplica la compensación asimétrica exclusiva al motor derecho."""
        magnitud = abs(v)
        if magnitud < 0.4:
            return v
        factor = self._compensacion(magnitud)
        return math.copysign(magnitud * factor - 0.2, v)

    def mover(self, vel_izq: float, vel_der: float):
        """Envía velocidades independientes aplicando compensación solo en movimiento lineal."""
        v_i = self._clamp(vel_izq)
        
        # Si tienen signos opuestos, están girando sobre su propio eje (spin).
        # Aplicar compensación asimétrica aquí suele desbalancear el giro.
        if v_i * vel_der < 0:
            v_d = self._clamp(vel_der)
        else:
            v_d = self._clamp(self._compensar_derecho(vel_der))
        
        self.robot.left_motor.value = v_i
        self.robot.right_motor.value = v_d

    def accion(self, tipo: str, velocidad: float = 0.5):
        """Ejecuta acciones predefinidas de movimiento."""
        acciones = {
            'avanzar': (velocidad, velocidad),
            'retroceder': (-velocidad, -velocidad),
            'izquierda': (-velocidad, velocidad),
            'derecha': (velocidad, -velocidad)
        }
        if tipo in acciones:
            v_i, v_d = acciones[tipo]
            self.mover(v_i, v_d)
        else:
            print(f"Acción '{tipo}' no reconocida.")

    def detener(self):
        """Detiene ambos motores inmediatamente."""
        self.robot.stop()

    def activar_driver(self):
        """Habilita el driver de potencia."""
        self.stby.on()

    def apagar_driver(self):
        """Detiene los motores y deshabilita el driver por seguridad."""
        self.detener()
        self.stby.off()
        print("Driver apagado y motores detenidos.")



def estimar_velocidades_chasis(self,vel_izq: float,vel_der: float,L_ancho_via_cm: float = 20.0):
    """
    Estima la velocidad lineal v (cm/s) y la velocidad angular w (rad/s)
    real del robot en base a los comandos normalizados y la compensación
    del polinomio.

    L_ancho_via_cm: Distancia física entre ruedas izquierdas y derechas (en cm).
    """

    # 1. Obtener los valores PWM ajustados por tu script
    v_i_pwm = self._clamp(vel_izq)
    v_d_pwm = self._clamp(self._compensar_derecho(vel_der))

# 2. Función auxiliar para mapear PWM -> Velocidad física (cm/s)
# Basado en la caracterización:
# v = 0.58 * (PWM %) + 15.53 (para PWM >= 40 %)
def pwm_a_velocidad_fisica(pwm_norm):
    magnitud = abs(pwm_norm)

    if magnitud < 0.4:
        return 0.0  # Zona muerta por debajo del 40 %

    pwm_porcentaje = magnitud * 100.0
    v_cm_s = 0.58 * pwm_porcentaje + 15.53

    return math.copysign(v_cm_s, pwm_norm)

# Velocidades de las ruedas en cm/s
v_L = pwm_a_velocidad_fisica(v_i_pwm)
v_R = pwm_a_velocidad_fisica(v_d_pwm)

# 3. Cinemática directa
v_centro = (v_R + v_L) / 2.0          # Velocidad lineal del robot (cm/s)
omega_chasis = (v_R - v_L) / L_ancho_via_cm  # Velocidad angular (rad/s)

return v_centro, omega_chasis

def Giros(Carrito,vel_izq,vel_der):
    '''Funcion compacta para pruebas de giro '''
    Carrito.mover(vel_izq,vel_der)

def menu_pruebas():
    print("\n--- MENÚ DE PRUEBAS DE GIRO ---")
    print("Opciones disponibles:")
    print("1. Giro suave")
    print("2. Giro medio")
    print("3. Giro asimetrico")
    print("4. Rotación (spin)")
    
    opcion = input("Selecciona una opción (1-4): ")

    velocidad_izq = float(input("Introduce la velocidad del motor izq. (0.0 a 1.0): "))
    velocidad_der = float(input("Introduce la velocidad del motor der. (0.0 a 1.0): "))
    duracion = float(input("Introduce la duración en segundos: "))

    acciones = {
        "1": "suave",
        "2": "medio",
        "3": "asimetrico",
        "4": "rotacion",
    }

    return acciones.get(opcion, None), velocidad_izq,velocidad_der, duracion

def ejecucion_giro(Carro, tipo, vel_izq, vel_der, duracion):
    print(f"Ejecutando giro: {tipo} a una proporción ({vel_izq}, {vel_der}) durante {duracion} segundos")
    Giros(Carro, vel_izq, vel_der)
    time.sleep(duracion)
    Carro.detener()


if __name__ == "__main__":
    pines_izq = (17, 27, 12)
    pines_der = (23, 22, 13)
    pin_stby = 24

    try:
        with Carro(pines_izq, pines_der, pin_stby) as carrito:
            while True:
                tipo, vel_izq, vel_der, duracion = menu_pruebas()
                if tipo is None:
                    print("Selecciona una opción válida, intenta de nuevo.")
                    continue

                ejecucion_giro(carrito, tipo, vel_izq, vel_der, duracion)

                seguir = input("\n¿Quieres hacer otra prueba? (s/n): ")
                if seguir.lower() != "s":
                    break    

    except KeyboardInterrupt:
        print("\nOperación interrumpida por el usuario.")
