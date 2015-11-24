import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
from scipy.integrate import odeint

# # --------------------------------------------------------------------------------------------------------------------
# # \ddot{x} + x = sin(2*t)
# N = 9
# T = 2*np.pi
# t = np.linspace(0, T, N+1)
# t = t[0:-1]
# f = np.sin(2*t)
# F = np.fft.fft(f)
# Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
# x0 = np.ones(N)
# xAnalytical = - np.sin(2*t) / 3
# def residual(x):
#     X = np.fft.fft(x)
#     ddx = np.fft.ifft(np.multiply(-Omega**2, X))
#     R = ddx + x - f
#     R = np.sum(np.abs((R**2)))
#     return R
#
# res = minimize(residual, x0)
# print(residual(res.x))
# plt.figure()
# plt.plot(t, res.x,
#          t, xAnalytical)
# plt.show()

# # --------------------------------------------------------------------------------------------------------------------
# # \ddot{x} + \dot{x} + x = sin(2*t)
# N = 19
# T = 2*np.pi
# # t = np.linspace(0, 2*np.pi, N+1)
# t = np.linspace(0, T, N+1)
# t = t[0:-1]
# f = np.sin(2*t)
# F = np.fft.fft(f)
# Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
# x0 = np.ones(N)
# # x0 = f
# xAnalytical = -0.23077 * np.sin(2*t) -0.15385 * np.cos(2*t)
# def residual(x):
#     X = np.fft.fft(x)
#     ddx = np.fft.ifft(np.multiply(-Omega**2, X))
#     dx = np.fft.ifft(np.multiply(1j * Omega, X))
#     R = ddx + dx + x - f
#     # R = np.sum(np.abs(np.real(R)))
#     R = np.sum(np.abs((R**2)))
#     return R
#
# print(residual(xAnalytical))
# # res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':100000})
# res = minimize(residual, x0)
# print(residual(res.x))
# plt.figure()
# plt.plot(t, res.x, 'k',
#          t, xAnalytical, 'ro')
# plt.legend(['FFt', 'Analytical'])
# plt.show()
# # print(res.jac)

# # --------------------------------------------------------------------------------------------------------------------
# # \ddot{x} + \dot{x} + x - x**3 = sin(2*t)
N = 199
T = 4*2*np.pi
t = np.linspace(0, T, N+1)
t = t[0:-1]
f = np.sin(2*t)
F = np.fft.fft(f)
Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
x0 = np.zeros(N)

def residual(x):
    X = np.fft.fft(x)
    ddx = np.fft.ifft(np.multiply(-Omega**2, X))
    dx = np.fft.ifft(np.multiply(1j * Omega, X))
    R = ddx + dx + x - x**3 - f
    R = np.sum(np.abs((R**2)))
    return R

# res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':1000000})
res = minimize(residual, x0)
xSol = res.x

# Numerical solution
def RHS(X, t=0.0):
    x1, x2 = X
    x1dot = x2
    x2dot = -x1 - x2 + x1**3 + np.sin(2*t)
    return [x1dot, x2dot]

ta = np.linspace(0.0, T, N)
sol = odeint(RHS, [0, 0], ta)
plt.figure()
plt.plot(t, res.x, 'k',
         ta, sol[:, 0], 'r')
plt.legend(['Harmonic Balance', 'Time integration'])
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.show()