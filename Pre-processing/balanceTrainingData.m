function [vectorX,vectorY] = balanceTrainingData(Xtrain,Ytrain)
% This function balances a training set by subsampling the examples from
% each class so that the number of examples for each class have the same
% number of elements, and this number is equal to the lowest number of
% examples among all classes.
%
% In the input and output matrices Xtrain and vector X, respectively, each
% row is an example and each column is a feature vector. The column vectors
% Ytrain and vectorY contain the labels for each example in Xtrain and
% vectorX, respectively.
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec


numExamplesPerClass = zeros(6,1);
for label = 1:6
    numExamplesPerClass(label) = sum(Ytrain == label);
end
minNumSamplesPerClass = min(numExamplesPerClass);

% Subsampling the examples from each class randomly with uniform
% probability
vectorX = [];
vectorY = [];
for label = 1:6
    [dummy, randIdx] = sort( rand(1, numExamplesPerClass(label)) );
    Xtrain_label = Xtrain(Ytrain == label, :);
    Ytrain_label = Ytrain(Ytrain == label);
    if label ~= 6
        vectorX = [vectorX; Xtrain_label( randIdx(1:minNumSamplesPerClass), : )];
        vectorY = [vectorY; Ytrain_label( randIdx(1:minNumSamplesPerClass) )];
    else
        vectorX = [vectorX; Xtrain_label( randIdx(1:3*minNumSamplesPerClass), : )];
        vectorY = [vectorY; Ytrain_label( randIdx(1:3*minNumSamplesPerClass) )];
    end
end

return