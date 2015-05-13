%clear workspace
clear all
close all

load fisheriris.mat
X = meas;
Y = species;

Y( strcmp('setosa',Y))={'1'};
Y( strcmp('versicolor',Y))={'2'};
Y( strcmp('virginica',Y))={'3'};

index = randsample(1:length(Y), 20);

Ytest = Y(index,:);
Ytest = str2double(Ytest);

Xtest = X(index,:);


Ytrain = str2double(Y);
Xtrain = X;

Ytrain(index,:) = [];
Xtrain(index,:) = [];
result = zeros(100,1);


for j=1:1
    [components, variances] = pca(Xtrain,1);

    sigma = 1;
    gamma = 1/(2*sigma^2);

    % %Libsvm library obtained online
    % % Libsvm options
    % % -s 0 : classification
    % % -t 2 : RBF (gaussian) kernel
    % % -g : gamma in the RBF kernel
    model = svmtrain(Ytrain, components, sprintf('-s 0 -t 2 -g %g', gamma));

    [predicted_label, accuracy, decision_values] = svmpredict(Ytest, Xtest, model);
    result(j,1)= accuracy(1,1);

end


