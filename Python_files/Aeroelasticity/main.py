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

N = 9
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
        if np.imag(x[ix]) > 1e-3 or np.imag(dx[ix]) > 1e-3:
            np.disp('You have a problem with imaginary numbers!')
            np.disp([np.imag(x[ix]), np.imag(dx[ix])])
        gs.wedge(theta=np.real(x[ix]), thetaDot=np.real(dx[ix]))
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

# plt.figure()
# plt.plot(t, xSol, 'k')
# plt.legend(['Harmonic Balance'])
# plt.xlabel('Time')
# plt.ylabel('Displacement')
# plt.show()