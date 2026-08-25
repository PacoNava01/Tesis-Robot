import time
from adafruit_servokit import ServoKit

# Configuración y constantes globales
NUM_SERVOS = 2
PULSE_WIDTH_RANGE = (405, 2500)
ACTUATION_RANGE = 180
LISTA_ANGULOS = [0, 90, 180, 90]


def init_servos(num_servos: int = NUM_SERVOS, pwr: tuple = PULSE_WIDTH_RANGE, act_range: int = ACTUATION_RANGE) -> ServoKit:
    """Inicializa la placa PCA9685 y configura los rangos de pulso y grados de los servos."""
    try:
        kit = ServoKit(channels=16)
        
        for i in range(num_servos):
            kit.servo[i].set_pulse_width_range(pwr[0], pwr[1])
            kit.servo[i].actuation_range = act_range
            
        print(f"{num_servos} servos inicializados correctamente.")
        return kit
    except Exception as e:
        print(f"Error al inicializar los servomotores: {e}")
        return None


def mover_servo(servo, angle: int):
    """Mueve un servo específico a un ángulo validando los límites."""
    angle_restringido = max(0, min(angle, ACTUATION_RANGE))
    servo.angle = angle_restringido
    print(f"Moviendo servo a: {angle_restringido}°")


def main():
    # Inicializar kit de servos
    kit = init_servos()
    if kit is None:
        return

    # Asignación semántica clara dentro del flujo principal
    servo_x = kit.servo[0]  # Eje horizontal
    servo_y = kit.servo[1]  # Eje vertical (opcional para uso futuro)

    print("Iniciando secuencia de prueba...")

    try:
        # Secuencia basada en lista de ángulos
        for angle in LISTA_ANGULOS:
            mover_servo(servo_x, angle)
            time.sleep(2.0)
            
        # Ejemplo opcional de movimiento fluido (barrido)
        print("Realizando barrido suave...")
        for angle in range(0, 181, 10):
            mover_servo(servo_x, angle)
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nSecuencia interrumpida por el usuario.")
    finally:
        # Posición de descanso segura al terminar
        print("Llevando el servo a posición neutral (90°)...")
        mover_servo(servo_x, 90)


if __name__ == "__main__":
    main()