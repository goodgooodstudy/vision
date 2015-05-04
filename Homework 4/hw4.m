%clear workspace
clear all
close all

s = -pi:0.1:pi;

k = 10;
r = 1/k;
s0=0;
alpha_x = r*cos(s+s0);
alpha_y = r*sin(s+s0);

alpha = [alpha_x', alpha_y'];
alphaMag = sqrt(alpha_x.^2 + alpha_y.^2);

figure;
scatter(alpha(:,1),alpha(:,2));
hold on;

x=r*cos(s0)+s*(-r*sin(s0))+ ((s.^2)/2)*(-r*cos(s0));
y=r*sin(s0)+s*(r*cos(s0))+ ((s.^2)/2)*(-r*sin(s0));
frenet = [x',y'];
frenetMeg = sqrt(x.^2+y.^2);

scatter(frenet(:,1),frenet(:,2));

magErr = abs (frenetMeg - alphaMag);
figure;
plot(magErr);
