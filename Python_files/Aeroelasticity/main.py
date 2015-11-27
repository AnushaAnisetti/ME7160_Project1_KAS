__author__ = 'koorosh'
import getShape as gs
import aerodynamic as aero
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
from scipy.integrate import odeint

gs.wedge(theta = 10, thetaDot = 1)
[Fx, Fy, Mz] = aero.calcLoad()

N = 5
T = 2*2*np.pi
t = np.linspace(0, T, N+1)
t = t[0:-1]
Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
x0 = np.zeros(N)

def residual(x):
    X = np.fft.fft(x)
    dx = np.fft.ifft(np.multiply(1j * Omega, X))
    ddx = np.fft.ifft(np.multiply(-Omega**2, X))
    f = np.zeros(N)
    for ix in range(0, len(x)):
        gs.wedge(theta = x[ix], thetaDot = dx[ix])
        [Fx, Fy, Mz] = aero.calcLoad()
        f[ix] = Mz
    Residual = ddx + 100 * x - Mz
    Residual = np.sum(np.abs((Residual**2)))
    return Residual

# res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':1000000})
res = minimize(residual, x0)
xSol = res.x
print(xSol)
print(residual(xSol))