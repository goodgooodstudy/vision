%clear workspace
clear all
close all

load gaussian.mat

XS = gaussian;
m=2;

[components, variances] = pca(XS,m);

figure;
scatter(XS(1,:),XS(2,:));
hold on;
quiver(components(1,:),components(2,:),norm(variances));
