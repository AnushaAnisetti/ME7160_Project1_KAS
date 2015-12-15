# Author: Koorosh Gobal
# Python code for 3.3
# -----------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
# -----------------------------------
epsilon = 1.0
mu = 1.0
alpha = 1.0
k = 1.0
omega = 2.0

N = 99
T = 2*2*np.pi
t = np.linspace(0, T, N+1)
t = t[0:-1]
Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
x0 = np.zeros(N)

# Harmonic Balance method
def residual(x):
    X = np.fft.fft(x)
    dx = np.fft.ifft(np.multiply(1j * Omega, X))
    ddx = np.fft.ifft(np.multiply(-Omega**2, X))
    Residual = ddx + x + epsilon * (2 * mu * dx + alpha * x**3
               + 2 * k * x * np.cos(omega * t)) - np.sin(2 * t)
    Residual = np.sum(np.abs((Residual**2)))
    return Residual
#
res = minimize(residual, x0, method='SLSQP')
xSol = res.x

# Numerical solution
def RHS(X, t=0.0):
    x1, x2 = X
    x1dot = x2
    x2dot = -x1 - epsilon * (2 * mu * x2 + alpha * x1**3
            + 2 * k * x1 * np.cos(omega * t)) + np.sin(2 * t)
    return [x1dot, x2dot]
#
ta = np.linspace(0.0, T, N)
sol = odeint(RHS, [0, 0], ta)
plt.figure()
plt.plot(t, res.x, 'k',
         ta, sol[:, 0], 'r--')
plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.show()
