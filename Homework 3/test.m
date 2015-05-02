%clear workspace
clear all
close all

A = [ 1 2 3 ; 1 2 3; 1 2 3; 1 2 3; 1 2 3];

indx = [2; 3; 1];

nonTrivial = zeros(size(A,1),size(A,2)-1);
for i = 2:size(indx,1)
    nonTrivial(:,i-1) = A(:,indx(i,1));
end