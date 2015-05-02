function [ map, densities, vals ] = diffmap( xs,sigma,t,m )
%compute pair-wise distance
pwdists2 = bsxfun(@plus, dot(xs,xs)', dot(xs,xs)) - 2*(xs'*xs);

%constract similiarity matrix, W
W = exp(-pwdists2./(2*sigma^2));

%get the row sum of W
rowSum = sum(W,2);

%make it diagonal matrix
D = diag(rowSum');

%constract M, really is just normalized W
M = (inv(D)) * W;

%desnities is the row sums of W normalized to be a probability vector
densities = sum(M,2);

%compute the first m+1 right eigenvectors of M
[eigVec,eigVal] = eigs(M, m+1);

%get val
vals = diag(eigVal);

%sort the vals to be descend order
[vals,indx] = sort(vals,'descend');
vals = vals(2:m+1,:);

%scale vals based on t
scale = vals.^t;

%rearrange the eigenvector based on sorted eigenvalues
nonTrivial = zeros(size(eigVec,1),size(eigVec,2)-1);
for i = 2:size(indx,1)
    nonTrivial(:,i-1) = eigVec(:,indx(i,1));
end

%add the scale to the map
map = bsxfun(@times,nonTrivial,scale');



end

