import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
from scipy.integrate import odeint

# # \ddot{x} + x = sin(2*t)
# N = 9
# t = np.linspace(0, 2*np.pi, N+1)
# t = t[0:-1]
# f = np.sin(2*t)
# F = np.fft.fft(f)
# Omega = np.fft.fftfreq(N, 1/N) + 0.00001
# x0 = np.ones(N)
# xAnalytical = - np.sin(2*t) / 3
# def residual(x):
#     X = np.fft.fft(x)
#     ddx = np.fft.ifft(np.multiply(-Omega**2, X))
#     R = ddx + x - f
#     R = np.sum(np.abs(R))
#     return R
#
# res = minimize(residual, x0)
# print(residual(res.x))
# plt.figure()
# plt.plot(t, res.x,
#          t, xAnalytical)
# plt.show()

# # \ddot{x} + \dot{x} + x = sin(2*t)
N = 19
t = np.linspace(0, 2*np.pi, N+1)
t = t[0:-1]
f = np.sin(2*t)
F = np.fft.fft(f)
Omega = np.fft.fftfreq(N, 1/N) + 0.00001
x0 = np.ones(N)
# x0 = f
xAnalytical = -0.23077 * np.sin(2*t) -0.15385 * np.cos(2*t)
def residual(x):
    X = np.fft.fft(x)
    ddx = np.fft.ifft(np.multiply(-Omega**2, X))
    dx = np.fft.ifft(np.multiply(1j * Omega, X))
    R = ddx + dx + x - f
    # R = np.sum(np.abs(np.real(R)))
    R = np.sum(np.abs((R**2)))
    return R

# res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':100000})
res = minimize(residual, x0)
print(residual(res.x))
plt.figure()
plt.plot(t, res.x, 'k',
         t, xAnalytical, 'ro')
plt.legend(['FFt', 'Analytical'])
plt.show()
print(res.jac)

# # \ddot{x} + x - x**3 = sin(2*t)
# N = 99
# t = np.linspace(0, 2*np.pi, N+1)
# t = t[0:-1]
# f = np.sin(2*t)
# F = np.fft.fft(f)
# Omega = np.fft.fftfreq(N, 1/N) + 0.00001
# x0 = np.ones(N)
#
# def residual(x):
#     X = np.fft.fft(x)
#     ddx = np.fft.ifft(np.multiply(-Omega**2, X))
#     # dx = np.fft.ifft(np.multiply(1j * Omega, X))
#     R = ddx + x - x**3 - f
#     R = np.sum(np.abs(R))
#     return R
#
# res = minimize(residual, x0, method='SLSQP')
# # res = fmin_slsqp(residual, x0, iter=100000)
# xSol = res.x
# print(residual(res.x))
# print(res)
#
#
# plt.figure()
# plt.plot(t, res.x)
# plt.show()