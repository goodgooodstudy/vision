%clear workspace
clear all
close all

load spiral.mat

%assign values to the parameters
xs = spiral;
sigma = 2;
m = 5;
t = 10;

%first plot the spiral with thetas
figure;
scatter(xs(1,:),xs(2,:),[],thetas);

[map, densities, vals] = diffmap(xs, sigma, t, m);

%scatter based on the diffusion map
x = ones(size(map,1),1);
figure;
scatter(map(:,1),x,[],thetas);
figure;
scatter(xs(1,:),xs(2,:),[],map(:,1));

