% This program checks the correctness of the computation of the gradient of
% an artificial neural network for multiclass classification

% Escuela Politecnica Nacional
% Marco E. Benalcázar Palacios
% marco.benalcazar@epn.edu.ec
clc;
close all;
clear all;
warning off all;

% Dimensions of training data
N = 1000; % Number of training examples
n = 35; % Dimension of the training examples
numLabels = 10; % Number of labels

% Architecture of the neural network
numNeuronsLayers = [n 25 15 12 10 5 numLabels];
transferFunctions{1} = 'none';
transferFunctions{2} = 'relu';
transferFunctions{3} = 'elu';
transferFunctions{4} = 'softplus';
transferFunctions{5} = 'tanh';
transferFunctions{6} = 'logsig';
transferFunctions{7} = 'softmax';
options.reluThresh = 1e-4;
options.lambda = 1e-1;

% Generating a table of frequencies randomly
dataX = randn(N, n);
dataY = randi([1 numLabels], N, 1);

% Generating the weights of the neural network randomly
mean = 0;
sigma = 0.15;
theta = [];
for i = 2:length(numNeuronsLayers)
    W = normrnd(mean, sigma, numNeuronsLayers(i), numNeuronsLayers(i - 1) + 1);
    theta = [theta; W(:)];
end

% Computing the gradients numerically
costFunction = @(t) softmaxNNCostFunction(dataX, dataY,...
                                            numNeuronsLayers,...
                                            t,...
                                            transferFunctions,...
                                            options);
numericalGradient = computeNumericalGradient( costFunction, theta );

% Computing the exact gradients
[dummyVar, analyticalGradient] = softmaxNNCostFunction(dataX, dataY,...
                                            numNeuronsLayers,...
                                            theta,...
                                            transferFunctions,...
                                            options);
                                        
% Comparing numerically computed gradients with those computed analytically
diff = norm(numericalGradient - analyticalGradient)/...
       norm(numericalGradient + analyticalGradient);
disp([numericalGradient analyticalGradient]);  
fprintf('\n difference = %d\n', diff);
% This value is usually less than 1e-7                                        