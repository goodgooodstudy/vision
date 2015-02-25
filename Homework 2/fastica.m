function [W, A] = fastica(data, numics)
    X = data;
    d = size(X,1);
    n = size(X,2);
    X_hat = zeros(size(X));

    %demean the sample X
    for i=1:d
        X_hat(i,:) = X(i,:) - mean2(X(i,:))*ones(size(X(i,:)));
    end

    %whiten the data
    [U,S,V] = svd(X_hat,'econ');
    X_tilda = sqrt(n-1) * (V.');
    W_tilda = zeros(numics,d);
    p=0;
    if numics == 1
        i=0;
        while 1
            fprintf('iteration %d\n ', i);

            wt = rand(d,1); %randomize the w
            
            a = X_tilda.'*wt;
            g = tanh(a);
            b = wt.'*X_tilda;
            g_prime = 1-tanh(b).^2;
            
            wt_plus = X_tilda*g - (g_prime*ones(n,1))*wt;
            wt_next = wt_plus / norm(wt_plus);
            display((abs(wt_next.'*wt)));
            i=i+1;
            if i > 1000 || (abs(wt_next.'*wt) > (1-10^-9))
                W_tilda = wt_next.';
                break;
            end
        end
    else
        for component = 1:numics
            i=0;
            while 1
                fprintf('%d iteration %d\n ',component, i);
                wt = rand(d,1);

                a = X_tilda.'*wt;          
                g = tanh(a);
                b = wt.'*X_tilda;
                g_prime = 1-tanh(b).^2;
                
                
                wt_plus = X_tilda*g - (g_prime*ones(n,1))*wt;
                wt_plus = wt_plus - p*wt_plus;
                wt_next = wt_plus / norm(wt_plus);
                i=i+1;
                if i > 1000 || (abs(wt_next.'*wt) > (1-10^-9))
                    p = p + wt_next * wt_next.';
                    W_tilda = wt_next.';
                    break;
                end 
            end
        end
    end
    
    %recover W and A from W_tilda
    M = ((1/sqrt(n-1))*U*S)^-1;
    W = W_tilda*M;
    A = M^-1 * W_tilda.';