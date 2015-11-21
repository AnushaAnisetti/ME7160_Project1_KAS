clc;
clear all;
close all;
format short g;
% ------------------------------------------------------------------------------------------------------------------------------ %
N = 59;
t = linspace(0, 2*pi, N+1)'; 
omega = 2 * pi / (t(end) - t(1));
t = t(1:end-1);
f = sin(2*t);

F = round(fft(f));

Omega = omega * [0,-1:-1:floor(-N/2),floor(N/2-1):-1:1]' + eps;

X = F ./ (1 - Omega.^2);
x = ifft(X)
figure,
plot(t, x,...
       t , -1/3 * sin(2 * t))