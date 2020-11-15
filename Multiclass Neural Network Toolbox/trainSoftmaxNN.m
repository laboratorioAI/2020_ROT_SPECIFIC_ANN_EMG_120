function weights = trainSoftmaxNN(dataX, dataY, numNeuronsLayers, transferFunctions, metaParameters)
%% This function trains a NEURAL NETWORK FOR MULTICLASS CLASSIFICATION by minimizing
%  the cross-entropy cost function
%
% Inputs:
% dataX                   [N n] matrix, where each row contains an observation
%                         X = (x_1, x_2,...,x_n)
%
% dataY                   [N 1] vector, where each row contains a label
%
% numNeuronsLayers        [1 L] vector [#_1, #_2,..., #_L], where #_1
%                         denotes the size of the input layer, #_2 denotes
%                         the size of the first hidden layer, #_3 denotes
%                         the size of the second hidden layer, and so on, and
%                         #_L = 1 denotes the size of the output layer
%
% transferFunctions       Cell containg the name of the transfer functions
%                         of each layer of the neural network. Options of transfer
%                         functions are:
%                         - none: input layer has no transfer functions
%                         - tanh: hyperbolic tangent
%                         - relu: rectified linear unit
%                         - logsig: logistic function
%
% metaParameters         structure containing additional settings for the
%                        neural network (e.g., rectified linear unit
%                        threshold, lambda, number of iterations, etc.)

% Escuela Politecnica Nacional
% Advanced Machine Learning
% Marco E. Benalcázar Palacios
% marco.benalcazar@epn.edu.ec

fprintf('Training an artificial neural network\n');

% Initializing the Neural Network Parameters Randomly
initialTheta = [];
for i = 2:length(numNeuronsLayers)
    r  = sqrt(6) / sqrt(numNeuronsLayers(i) + numNeuronsLayers(i - 1) + 1);
     rng('default')
    W = rand(numNeuronsLayers(i), numNeuronsLayers(i - 1) + 1) * 2 * r - r;
    % mean = 0;
    % sigma = 0.01;
    %     W = normrnd(mean, sigma, numNeuronsLayers(i), numNeuronsLayers(i - 1) + 1);
    initialTheta = [initialTheta; W(:)];
end

% Unrolling parameters
options = optimset('MaxIter', metaParameters.numIterations);
costFunction = @(t) softmaxNNCostFunction(dataX, dataY,...
    numNeuronsLayers,...
    t,...
    transferFunctions,...
    metaParameters);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[theta, cost, iterations] = fmincg(costFunction, initialTheta, options);

% Plotting the error curve
figure;
try
    plot(1:iterations,cost,'*r','LineWidth',2);
catch
    plot(1:length(cost),cost,'*r','LineWidth',2);
end
xlabel('Epoch number');
ylabel('Cost value');
grid on;
drawnow;
ylim([0 max(cost)*1.05]);
% Reshaping the weight matrices
numLayers = length(numNeuronsLayers);
endPoint = 0;
for i = 2:numLayers
    numRows = numNeuronsLayers(i);
    numCols = numNeuronsLayers(i - 1) + 1;
    numWeights = numRows*numCols;
    startPoint = endPoint + 1;
    endPoint = endPoint + numWeights;
    weights{i - 1} = reshape(theta(startPoint:endPoint), numRows, numCols);
end

% Computing the training error
[dummyVar, A] = forwardPropagation(dataX, weights, transferFunctions, metaParameters);
P = A{end};
[dummyVar, predictedLabels] = max(P, [], 2);
trainingAccuracy = 100*sum(predictedLabels == dataY)/length(dataY);
fprintf('Training Accuracy of the NEURAL NETWORK: %1.2f %%\n\n', trainingAccuracy);
return