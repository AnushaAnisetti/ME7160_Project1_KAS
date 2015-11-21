clc;
clear all;
close all;
format short g;
% ------------------------------------------------------------------------------------------------------------------------------ %
N = 59; % Number of sample points in time
t = linspace(0, 2*pi, N+1)';  % Time
omega = 2 * pi / (t(end) - t(1)); % delta f
t = t(1:end-1);
f = sin(2*t) + cos(3*t); % Forcing functions

F = round(fft(f));

Omega = omega * [0,-1:-1:floor(-N/2),floor(N/2-1):-1:1]' + eps;

X = F ./ (1 - Omega.^2); % Solution in frequency domain
x = ifft(X)
figure,
plot(t, x, 'k', ...
       t , -1/3 * sin(2 * t) - 1/8 * cos(3*t), 'r')
xlabel('Time')
ylabel('Displacement')
legend('FFT Solution', 'Analytical Solution')