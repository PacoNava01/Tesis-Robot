from gpiozero import Robot, Motor, OutputDevice
import math
import time


class Carro:
    """
    Controlador para un robot diferencial con:

    - Compensación polinómica del motor derecho.
    - Gestión del pin STBY del driver.
    - Estimación cinemática de velocidades reales.
    """

    COEFS = {
        'a': 0.9356779218031275,
        'b': -1.864449111530371,
        'c': 1.1723900277729624,
        'd': -0.2070794160209165
    }

    def __init__(
        self,
        left_pins: tuple,
        right_pins: tuple,
        stby_pin: int,
        invertir_stby: bool = False
    ):
        """
        Parameters
        ----------
        left_pins : (IN1, IN2, PWM)
        right_pins : (IN1, IN2, PWM)
        stby_pin : Pin STBY del driver
        invertir_stby : True si STBY es activo en bajo
        """

        self.stby = OutputDevice(
            stby_pin,
            active_high=not invertir_stby
        )

        self.activar_driver()

        motor_izq = Motor(
            forward=left_pins[0],
            backward=left_pins[1],
            enable=left_pins[2]
        )

        motor_der = Motor(
            forward=right_pins[0],
            backward=right_pins[1],
            enable=right_pins[2]
        )

        self.robot = Robot(
            left=motor_izq,
            right=motor_der
        )

    # ============================================================
    # Context Manager
    # ============================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.apagar_driver()

    # ============================================================
    # Métodos auxiliares
    # ============================================================

    @staticmethod
    def _clamp(
        valor: float,
        minimo: float = -1.0,
        maximo: float = 1.0
    ) -> float:
        """Limita un valor al rango permitido."""
        return max(minimo, min(maximo, valor))

    def _compensacion(self, pwm: float) -> float:
        """
        Obtiene el factor de corrección polinómico
        para el motor derecho.
        """

        a, b, c, d = (
            self.COEFS[k]
            for k in ('a', 'b', 'c', 'd')
        )

        x = max(0.4, min(1.0, pwm))

        error = (
            a * x**3 +
            b * x**2 +
            c * x +
            d
        )

        return 1 - error

    def _compensar_derecho(self, velocidad: float) -> float:
        """
        Aplica compensación únicamente al motor derecho.
        """

        magnitud = abs(velocidad)

        if magnitud < 0.4:
            return velocidad

        factor = self._compensacion(magnitud)

        return math.copysign(
            magnitud * factor - 0.2,
            velocidad
        )

    # ============================================================
    # Cinemática
    # ============================================================

    def _pwm_a_velocidad_fisica(self, pwm_norm: float) -> float:
        """
        Convierte PWM normalizado [-1,1] a velocidad física [cm/s].

        Modelo experimental:
            v = 0.58(PWM %) + 15.53

        válido para PWM >= 40 %
        """

        magnitud = abs(pwm_norm)

        if magnitud < 0.4:
            return 0.0

        pwm_porcentaje = magnitud * 100.0

        velocidad = (
            0.58 * pwm_porcentaje +
            15.53
        )

        return math.copysign(
            velocidad,
            pwm_norm
        )

    def estimar_velocidades_chasis(
        self,
        vel_izq: float,
        vel_der: float,
        ancho_via_cm: float = 20.0
    ):
        """
        Estima velocidades reales del robot.

        Returns
        -------
        (v, omega)

        v      : velocidad lineal [cm/s]
        omega  : velocidad angular [rad/s]
        """

        v_i_pwm = self._clamp(vel_izq)

        if vel_izq * vel_der < 0:
            v_d_pwm = self._clamp(vel_der)
        else:
            v_d_pwm = self._clamp(
                self._compensar_derecho(vel_der)
            )

        v_L = self._pwm_a_velocidad_fisica(v_i_pwm)
        v_R = self._pwm_a_velocidad_fisica(v_d_pwm)

        v_centro = (v_L + v_R) / 2.0

        omega = (
            (v_R - v_L) /
            ancho_via_cm
        )

        return v_centro, omega

    # ============================================================
    # Movimiento
    # ============================================================

    def mover(
        self,
        vel_izq: float,
        vel_der: float
    ):
        """
        Envía velocidades independientes a los motores.
        """

        v_i = self._clamp(vel_izq)

        # Rotación sobre el eje
        if vel_izq * vel_der < 0:
            v_d = self._clamp(vel_der)
        else:
            v_d = self._clamp(
                self._compensar_derecho(vel_der)
            )

        self.robot.left_motor.value = v_i
        self.robot.right_motor.value = v_d

    def accion(
        self,
        tipo: str,
        velocidad: float = 0.5
    ):
        """
        Acciones predefinidas.
        """

        acciones = {
            "avanzar": (velocidad, velocidad),
            "retroceder": (-velocidad, -velocidad),
            "izquierda": (-velocidad, velocidad),
            "derecha": (velocidad, -velocidad)
        }

        if tipo in acciones:
            vi, vd = acciones[tipo]
            self.mover(vi, vd)
        else:
            print(f"Acción '{tipo}' no reconocida.")

    def detener(self):
        """Detención inmediata."""
        self.robot.stop()

    # ============================================================
    # Driver
    # ============================================================

    def activar_driver(self):
        """Habilita el driver."""
        self.stby.on()

    def apagar_driver(self):
        """Apaga el driver de forma segura."""
        self.detener()
        self.stby.off()
        print("\nDriver apagado. Motores detenidos.")


# ============================================================
# Utilidades de interfaz
# ============================================================

def leer_velocidad(mensaje):
    while True:
        try:
            valor = float(input(mensaje))

            if -1.0 <= valor <= 1.0:
                return valor

            print("Valor fuera de rango [-1.0, 1.0]")

        except ValueError:
            print("Introduce un número válido.")


def leer_flotante_positivo(mensaje):
    while True:
        try:
            valor = float(input(mensaje))

            if valor > 0:
                return valor

            print("Introduce un valor positivo.")

        except ValueError:
            print("Entrada no válida.")


def menu_pruebas():
    print("\n========== MENÚ DE GIROS ==========")
    print("1. Giro suave")
    print("2. Giro medio")
    print("3. Giro asimétrico")
    print("4. Rotación (spin)")
    print("===================================")

    opcion = input("Selecciona opción (1-4): ")

    acciones = {
        "1": "suave",
        "2": "medio",
        "3": "asimetrico",
        "4": "rotacion"
    }

    tipo = acciones.get(opcion)

    if tipo is None:
        return None, None, None, None

    vel_izq = leer_velocidad(
        "Velocidad izquierda [-1.0,1.0]: "
    )

    vel_der = leer_velocidad(
        "Velocidad derecha [-1.0,1.0]: "
    )

    duracion = leer_flotante_positivo(
        "Duración [s]: "
    )

    return (
        tipo,
        vel_izq,
        vel_der,
        duracion
    )


def ejecucion_giro(
    carro,
    tipo,
    vel_izq,
    vel_der,
    duracion
):
    """
    Ejecuta un movimiento de prueba.
    """

    v, omega = carro.estimar_velocidades_chasis(
        vel_izq,
        vel_der
    )

    print("\n--------------------------------")
    print(f"Tipo de giro : {tipo}")
    print(f"Motor Izq    : {vel_izq:.2f}")
    print(f"Motor Der    : {vel_der:.2f}")
    print(f"Duración     : {duracion:.2f} s")
    print(f"Vel. lineal  : {v:.2f} cm/s")
    print(f"Vel. angular : {omega:.2f} rad/s")
    print("--------------------------------")

    carro.mover(
        vel_izq,
        vel_der
    )

    time.sleep(duracion)

    carro.detener()


# ============================================================
# Programa principal
# ============================================================

if __name__ == "__main__":

    PINES_IZQ = (17, 27, 12)
    PINES_DER = (23, 22, 13)
    PIN_STBY = 24

    try:
        with Carro(
            PINES_IZQ,
            PINES_DER,
            PIN_STBY
        ) as carrito:

            while True:

                tipo, vel_izq, vel_der, duracion = (
                    menu_pruebas()
                )

                if tipo is None:
                    print(
                        "\nOpción inválida. "
                        "Intenta nuevamente."
                    )
                    continue

                ejecucion_giro(
                    carrito,
                    tipo,
                    vel_izq,
                    vel_der,
                    duracion
                )

                seguir = input(
                    "\n¿Realizar otra prueba? (s/n): "
                )

                if seguir.lower() != "s":
                    break

    except KeyboardInterrupt:
        print("\nOperación interrumpida por el usuario.")
