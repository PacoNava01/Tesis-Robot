import numpy as np
import time

#-----Clase-----
class PID:
    def __init__(self,kP,kI,kD):
        self.kP,self.kI,self.kD = kP,kI,kD
        self.last_error = 0
        self.integral = 0 
        self.last_time = time.time()

    def update(self,error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: return 0

        #Proporcional 
        P = self.kP * error 

        # Integral (antiwindup)
        self.integral += error * dt
        self.integral = max(-10, min(10, self.integral))
        I = self.kI * self.integral
        
        # Derivativo
        D = self.kD * (error - self.last_error) / dt

        self.last_error = error
        self.last_time = now
        return P + I + D
