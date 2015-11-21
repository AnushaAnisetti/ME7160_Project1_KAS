clc;
clear all;
close all;
format short g;
% ----------------------------------------------------------------------- %
% N = 9;
% t = linspace(0, 2*pi, N + 1)'; t = t(1:N);
% f = sin(2 * t);
% k = (1:N)';
% 
% F = fft(f);
% X = F ./ (1 - 4 * pi^2 * ((k - 1)./(N)).^2);
% x = ifft(X);
% xAnal1 = -1/3 * sin(2*t) + 5 / 3 * sin(t);
% xAnal2 = -1/3 * sin(2*t);
% 
% figure,
% plot(t, x, 'k',...
%      t, xAnal1, 'k--',...
%      t, xAnal2, 'k-.',...
%      'linewidth', 3)
% legend('fft', 'H + P', 'P')
% xlabel('Time','fontsize',20)
% ylabel('Displacement','fontsize',20)
% set(gca,'fontsize',20)

% nt = 1000;
% x1 = zeros(nt,1);
% x2 = zeros(nt,1); x2(1) = 1;
% t = linspace(0, 10, nt);
% dt = t(2) - t(1);
% for it=1:nt-1
%     x1(it+1) = x2(it) * dt + x1(it);
%     x2(it+1) = (-x1(it) + sin(2*it*dt)) * dt + x2(it);
% end
% 
% figure,
% plot(t, x1,...
%      t, -1/3 * sin(2*t) + 5 / 3 * sin(t), 'o')

N = 20;
t = linspace(0, 2*pi, N+1)'; t = t(1:end-1);

T = t(2) - t(1);
fs = 1/T

x = cos(2*t);
X = round(fft(x));
f = [0,1:ceil(N/2),(ceil(N/2)-1):-1:1]';
[X, f]