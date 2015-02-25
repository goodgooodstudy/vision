function [W, A] = fastica(data, numics)
    X = data;
    d = size(X,1);
    n = size(X,2);
    mean = zeros(d,1);
    X_hat = zeros(size(X));

    %demean the sample X
    for i=1:d
        mean(i,1)=sum(X(i,:))/n;
        X_hat(i,:) = X(i,:) - mean(i,1);
    end

    %whiten the data
    [U,S,V] = svd(X_hat,'econ');
    X_tilda = sqrt(n-1) * (V.');

    W_tilda = zeros(numics,d);

    %find numics rows of wt_p for the whitened data
    for p = 1:numics
        wt_p = rand(d,1); %randomize the wt_p
        i=0;
        while 1
            wt_prev = wt_p;
            a = wt_p.'*X_tilda;
            b = X_tilda.'*wt_p;
            c = 1-tanh(a).^2;

            wt_p = (1/n)*X_tilda*tanh(b)-(1/n)*(c*ones(n,1))*wt_p;
            
            %update W with row order
            for j=1:(p-1)
                wt_p = wt_p -  W_tilda(j,:).'*W_tilda(j,:)*wt_p;
            end

            wt_p = wt_p/norm(wt_p);

            if i > 1000 || (abs(wt_p.'*wt_prev) > (1-10^-9))
                break; 
            end
            i=i+1;
        end
        W_tilda(p,:) = wt_p.';
    end

    %recover W and A from W_tilda
    M = ((1/sqrt(n-1))*U*S)^-1;
    W = W_tilda*M;
    A = M^-1 * W_tilda.';