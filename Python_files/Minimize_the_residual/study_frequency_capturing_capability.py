__author__ = 'koorosh'
# # Study for which frequencies can be captured
T = 8 * np.pi
N = 9
t = np.linspace(0, T, N+1)
t = t[:-1]
n = np.fft.fftfreq(N, 1/N)
omega0 = 2*np.pi / T
omega = omega0 * n

a = 0.5
x = np.sin(a*t)
dxdt = a*np.cos(a*t)

X = np.fft.fft(x)
DxDt = np.fft.ifft(1j * np.multiply(omega, X))

print(n)
print(np.max(np.abs(np.real(DxDt - dxdt))))