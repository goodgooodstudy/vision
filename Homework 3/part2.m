%clear workspace
clear all
close all

load gaussian.mat

XS = gaussian;

[components, variances] = pca(XS,2);

figure;
scatter(XS(1,:),XS(2,:));
hold on;
quiver(components(1,:),components(2,:),5);
