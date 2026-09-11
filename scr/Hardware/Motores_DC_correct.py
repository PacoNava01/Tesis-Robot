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


if __name__ == "__main__":
    # Definición de pines (GPIOs físicos de la Raspberry Pi)
    pines_izq = (17, 27, 12)  # Forward, Backward, Enable (PWM)
    pines_der = (23, 22, 13)  # Forward, Backward, Enable (PWM)
    pin_stby = 24

    # Uso seguro mediante Context Manager ('with')
    try:
        with Carro(pines_izq, pines_der, pin_stby) as carrito:
            #print("Avanzando durante 2 segundos...")
            #carrito.accion('derecha', velocidad=0.5)
            #carrito.accion('avanzar', velocidad=0.2)
            #time.sleep(1.0)

            #print("Girando a la izquierda...")
            #carrito.accion('izquierda', velocidad=0.5)
            #time.sleep(1.0)

            #print("Deteniendo...")
            #carrito.detener()
            #time.sleep(1.0)
            
            carrito.mover(0.6,-0.6)
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nOperación interrumpida por el usuario.")