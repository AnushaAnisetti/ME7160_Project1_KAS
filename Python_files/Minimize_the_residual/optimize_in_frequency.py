__author__ = 'koorosh gobal'
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# # --------------------------------------------------------------------------------------------------------------------
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)
linewidth = 9.0
markersize = 20
# # --------------------------------------------------------------------------------------------------------------------
# # \ddot{x} + \dot{x} + x = sin(2*t)

N = 4
n = np.fft.fftfreq(N, 1/N)
f0 = 1
omega = 2 * np.pi * f0 * n

t = np.linspace(0, N*f0, N+1)
t = t[:-1]
print(np.sin(t))
print(np.fft.ifft(1j * np.multiply(omega, np.fft.fft(-np.cos(t)))))

# N = 19
# T = 2*np.pi
# n = np.fft.fftfreq(N, 1/N)
# f0 = [1]
#
# x0 = np.zeros(N)
# x0 = np.concatenate((x0, f0))
# # x0 = f
# def residual(x):
#     x = x[:-1]
#     f0 = x[-1]
#     omega = 2 * np.pi * f0 * n
#     print(x)
#     t = np.linspace(0, np.multiply(N, f0), N+1)
#     t = t[0:-1]
#     f = np.sin(2*t)
#
#     X = np.fft.fft(x)
#     ddx = np.fft.ifft(np.multiply(-omega**2, X))
#     dx = np.fft.ifft(np.multiply(1j * omega, X))
#     R = ddx + dx + x - f
#     # R = np.sum(np.abs(np.real(R)))
#     R = np.sum(np.abs((R**2)))
#     return R
#
# # print(residual(xAnalytical))
# # res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':100000})
# res = minimize(residual, x0)
#
# xFFTsol = res.x[:-1]
# f0 = res.x[-1]
# t = np.linspace(0, np.multiply(N, f0), N+1)
# t = t[0:-1]
# f = np.sin(2*t)
# xAnalytical = -0.23077 * np.sin(2*t) -0.15385 * np.cos(2*t)
# print(res.x)
# # print(residual(res.x))
# # # plt.figure(figsize=(30,15))
# plt.figure()
# plt.plot(t, xFFTsol, 'k',
#          t, xAnalytical, 'r--o',
#          lw=linewidth, ms=markersize)
# plt.legend(['FFt', 'Analytical'])
# plt.xlabel('Time')
# plt.ylabel('Displacement')
# plt.savefig('1N19.eps', format='eps', dpi=1000, bbox_inches='tight')
# plt.show()
# print(res.jac)

