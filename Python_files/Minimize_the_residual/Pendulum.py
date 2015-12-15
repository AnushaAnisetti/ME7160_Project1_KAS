import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
from scipy.integrate import odeint
# # -------------------------------------------------------------------------------------------------------------------
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)
linewidth = 4.0
markersize = 10
# # -------------------------------------------------------------------------------------------------------------------
# # 2DOF Mass-Spring-Damper System
# # Define system properties
m1 = 2
m2 = 1
l1 = 1
l2 = 2
g=9.81


N = 99
T = 5*2*np.pi
t = np.linspace(0, T, N+1)
t = t[0:-1]
p = np.sin(2*t)
Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
x0 = np.zeros(N*2)

def residual(x):
    x1 = x[:N]
    x2 = x[N:]

    X1 = np.fft.fft(x1)
    X2 = np.fft.fft(x2)
    dx1 = np.fft.ifft(np.multiply(1j * Omega, X1))
    dx2 = np.fft.ifft(np.multiply(1j * Omega, X2))
    ddx1 = np.fft.ifft(np.multiply(-Omega**2, X1))
    ddx2 = np.fft.ifft(np.multiply(-Omega**2, X2))

    R1 = (m1+m2)*l1*ddx1+m2*l2*np.cos(x[:N]-x[N:])*ddx2+m2*l2*np.sin(x[:N]-x[N:])*dx2**2+g*(m1+m2)*np.sin(x[:N])
    R2 = m2*l2*ddx2+m2*l1*np.cos(x[:N]-x[N:])*ddx1-m2*l1*np.sin(x[:N]-x[N:])*dx2**2+g*(m2)*np.sin(x[N:]) - p
    Residual = R1**2 + R2**2
    Residual = np.sum(np.abs((Residual)))
    return Residual

# # res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':1000000})
res = minimize(residual, x0)
print(residual(res.x))
xSol = res.x
xSol1 = xSol[:N]
xSol2 = xSol[N:]

# Numerical solution
def RHS(X, t=0.0):
     x11, x12, x21, x22 = X
     x11dot = x12
     x21dot = x22
     a = (m1+m2)*l1
     b = m2*l2*np.cos(x11-x21)
     c = m2*l1*np.cos(x11-x21)
     d = m2*l2
     e = -m2*l2*x22**2*np.sin(x11-x21)-g*(m1+m2)*np.sin(x11)
     f = m2*l1*x12**2*np.sin(x11-x21)-m2*g*np.sin(x21)+p
     x12dot = (e*d-b*f)/(a*d-c*b)
     x22dot = (a*f-c*e)/(a*d-c*b)
     return [x11dot, x12dot, x21dot, x22dot]
#
ta = np.linspace(0.0, T, 20*N)
sol = odeint(RHS, [0, 0, 0, 0], ta)
print(sol)
# # plt.figure(figsize=(30,15))
plt.figure()
plt.plot(t, xSol1, 'k',
         ta, sol[:, 0], 'r--',
         lw=linewidth, ms=markersize)
plt.legend(['FFt', 'Analytical'])
plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
plt.title('x1')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.show()
#
plt.figure()
plt.plot(t, xSol2, 'k',
         ta, sol[:, 2], 'r--',
         lw=linewidth, ms=markersize)
plt.legend(['FFt', 'Analytical'])
plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
plt.title('x2')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.show()