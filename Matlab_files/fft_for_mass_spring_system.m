clc;
clear all;
close all;
format short g;
% ----------------------------------------------------------------------- %
N = 9;
t = linspace(0, 2*pi, N + 1)'; t = t(1:N);
f = sin(2 * t);
k = (1:N)';

F = fft(f);
X = F ./ (1 - 4 * pi^2 * ((k - 1)./(N)).^2);
x = ifft(X);
xAnal1 = -1/3 * sin(2*t) + 5 / 3 * sin(t);
xAnal2 = -1/3 * sin(2*t);

figure,
plot(t, x, 'k',...
     t, xAnal1, 'k--',...
     t, xAnal2, 'k-.',...
     'linewidth', 3)
legend('fft', 'H + P', 'P')
xlabel('Time','fontsize',20)
ylabel('Displacement','fontsize',20)
set(gca,'fontsize',20)

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

% N = 500;
% t = linspace(0, 2*pi, N + 1)'; t = t(1:N);
% f = sin(2 * t);
% k = (-N/2:N/2)';
% 
% trapz(t, sin(t).*sin(t))
% 
% F = zeros(N, 1);
% for ik=1:length(k)
%     F(ik) = trapz(t, f .* exp(-2 * pi * i * k(ik) * t));
% end
% 
% X = F ./ (1 - 4 * pi^2 * k.^2);
% 
% x = zeros(N, 1);
% for it=1:length(t)
%     x(it) = trapz(k, X .* exp(2*pi*i*k*t(it)));
% end
% 
% figure,
% plot(t, real(x))