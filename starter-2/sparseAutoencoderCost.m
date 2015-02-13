function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) given
% hiddenSize: the number of hidden units (probably 25) given
% lambda: weight decay parameter, given
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks
%                           like a lower-case "p"). given
% beta: weight of sparsity penalty term, given
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0; %FIXME: what is the cost ?
phat = zeros([hiddenSize,1]);
dW1 = zeros(size(W1)); 
dW2 = zeros(size(W2));
db1 = zeros(size(b1)); 
db2 = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
    m = size(data,2);
    for dataSetIdx = 1:m
        %hidden layer
        %---------------------------------step 0---------------------------------
        %calculate phat first
        %------------------------------------------------------------------------
        z2 = W1 * data(:,dataSetIdx) + b1;
        a2 = sigmoid(z2); %activation for layer 2
        phat = phat + a2;
    end
    
    phat = (1/m) * phat;
    
    for dataSetIdx = 1:m
        %hidden layer
        %---------------------------------step 1---------------------------------
        %perform the forward passing, computing the activations for each layer up
        %to the output layer L3
        %------------------------------------------------------------------------
        z2 = W1 * data(:,dataSetIdx) + b1;
        a2 = sigmoid(z2); %activation for layer 2
        %output layer
        z3 = W2 * a2 + b2;
        a3 = sigmoid(z3); %a3 = h(w,b; x), activation (output) for layer 3

        %---------------------------------step 1.5-------------------------------
        %get the cost of our cost function
        %------------------------------------------------------------------------  
        cost = cost + (1/2)*norm(data(:,dataSetIdx) - a3)^2;

        %---------------------------------step 2---------------------------------
        %for each output unit a3(:,i), we find error term, err, derivative over
        %LMS
        %------------------------------------------------------------------------
        errl3 = -1*( data(:,dataSetIdx) - a3) .* (a3 .* (1-a3));

        %---------------------------------step 3---------------------------------
        %calculate error term for hidden layer, layer 2
        %------------------------------------------------------------------------
        errl2 = (W2.' * errl3 + beta * (-sparsityParam ./ phat + (1-sparsityParam)./(1-phat))) .* (a2 .* (1-a2));

        %---------------------------------step 4---------------------------------
        %compute the desired partial derivatives
        %------------------------------------------------------------------------
        jW2 = errl3 * a2.';
        jW1 = errl2 * data(:,dataSetIdx).'; %input x is a1
        jb2 = errl3;
        jb1 = errl2;
        %---------------------------------step 5---------------------------------
        %update gradient
        %------------------------------------------------------------------------
        dW1 = dW1 + jW1;
        dW2 = dW2 + jW2;
        db1 = db1 + jb1;
        db2 = db2 + jb2;
    end

    %---------------------------------step 6---------------------------------
    %do deriative of J(W,b), meaning i do 1/m * dW1
    %------------------------------------------------------------------------
    W1grad = (1/m) * dW1 + lambda * W1;
    W2grad = (1/m) * dW2 + lambda * W2;
    b1grad = (1/m) * db1;
    b2grad = (1/m) * db2;
    cost = (1/m) * cost;
    
    term1 = log(sparsityParam./phat);
    term2 = log((1-sparsityParam)./(1-phat));
    KL = sparsityParam * term1 + (1-sparsityParam)*term2;
    cost = cost + (lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2))) + beta * sum(KL);
    %-------------------------------------------------------------------
    % After computing the cost and gradient, we will convert the gradients back
    % to a vector format (suitable for minFunc).  Specifically, we will unroll
    % your gradient matrices into a vector.

    grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end




