__author__ = 'koorosh'
import getShape as gs
import aerodynamic as aero
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
from scipy.integrate import odeint

gs.wedge(theta = 0, thetaDot = 5)
[Fx, Fy, Mz] = aero.calcLoad()

T = 2*np.pi
N = 5
t = np.linspace(0, T, N+1)
t = t[:-1]
n = np.fft.fftfreq(N, 1/N)
omega0 = 2*np.pi / T
Omega = omega0 * n

x0 = np.zeros(N)
x0 = np.concatenate((x0, [T]))

def residual(x):
    T = x[-1]
    x = x[:-1]
    omega0 = 2*np.pi / T
    Omega = omega0 * n
    t = np.linspace(0, T, N+1)
    t = t[0:-1]

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

res = minimize(residual, x0, method='SLSQP', options={'maxiter':10000000})
# res = minimize(residual, x0)
xSol = res.x[:-1]
T = res.x[-1]
print(xSol)
print(residual(xSol))
np.savetxt('HB_X.txt', xSol, fmt='%2.2f')
np.savetxt('HB_t.txt', t, fmt='%2.2f')

# Numerical solution
def RHS(X, t=0.0):
    x1, x2 = X
    gs.wedge(theta=x1, thetaDot=x2)
    [Fx, Fy, Mz] = aero.calcLoad()
    x1dot = x2
    x2dot = -100*x1 + Mz
    return [x1dot, x2dot]

ta = np.linspace(0.0, 1*T, 50*N)
sol = odeint(RHS, [0, 0], ta)
np.savetxt('ND_X.txt', sol[:, 0], fmt='%2.2f')
np.savetxt('ND_Xdot.txt', sol[:, 1], fmt='%2.2f')
np.savetxt('ND_T.txt', ta, fmt='%2.2f')

plt.figure()
plt.plot(t, xSol, 'k',
         ta, sol[:, 0], 'ro')
plt.legend(['Harmonic Balance', 'Numerical Differentiation'])
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.show()