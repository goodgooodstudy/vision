function [ components, variances ] = pca(XS, m)
    %center XS
    for i=1:size(XS,1)
        xMean = mean(XS(i,:));
        display(xMean);
        XC(i,:) = XS(i,:) -  xMean;
    end
    
    %calculate SVD with m sigular values and principle components
    [U,S,V] = svds(XC,m);
    
    %convert singular values retured in step2 into variance estimates
    variances = (diag(S).^2)./(size(XC,2)-1);
    components = U;
end

