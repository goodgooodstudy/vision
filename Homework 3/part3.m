%clear workspace
clear all
close all

load spiral.mat

xs = spiral;
sigma = 1;
m = 2;
t = 10;


figure;
scatter(xs(1,:),xs(2,:),[],thetas);

%compute pair-wise distance
pwdists2 = bsxfun(@plus, dot(xs,xs)', dot(xs,xs)) - 2*(xs'*xs);

%constract similiarity matrix, W
W = exp(-pwdists2./(2*sigma^2));

%get the row sum of W
rowSum = sum(W,2);

%make it diagonal matrix
D = diag(rowSum');

%constract M
M = (inv(D)) * W;

%compute the first m+1 right eigenvectors of M
[eigVec,eigVal] = eigs(M, m+1);

%get val
vals = diag(eigVal);
vals = vals(2:m+1,:);

%scale vals based on t
scale = vals.^t;

%add the scale to the map (first non-trivial eigevvector
map = bsxfun(@times,eigVec(:,2:m+1),scale');

x = ones(size(map,1),1);
figure;
scatter(map(:,1),x,[],thetas);



