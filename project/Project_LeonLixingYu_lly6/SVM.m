%clear workspace
clear all
close all

load ovariancancer.mat
X = obs;
Y = grp;

Y( strcmp('Cancer',Y))={'1'};
Y( strcmp('Normal',Y))={'-1'};
%Y( strcmp('virginica',Y))={'3'};

index = randsample(1:length(Y), 30);

Ytest = Y(index,:);
Ytest = str2double(Ytest);

Xtest = X(index,:);


Ytrain = str2double(Y);
Xtrain = X;

Ytrain(index,:) = [];
Xtrain(index,:) = [];

sigma = 10;
gamma = 1/(2*sigma^2);

% %Libsvm library obtained online
% % Libsvm options
% % -s 0 : classification
% % -t 2 : RBF (gaussian) kernel
% % -g : gamma in the RBF kernel
result = zeros(100,1);
for j=1:1
    model = svmtrain(Ytrain, double(Xtrain), sprintf('-s 0 -t 2 -g %g', gamma));

    [predicted_label, accuracy, decision_values] = svmpredict(Ytest, double(Xtest), model);
    
    result(j,1)= accuracy(1,1);

end
