clc;
close all;
clear all;
% This script contains the code of the algorithm presented in the paper
% entitled "A User-Specific Hand Gesture Recognition Model Based on
% Feed-Forward Neural Networks, EMGs and Correction of Sensor Orientation"
% This code has been developed using Matlab 2020a
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec
% April 15, 2020

%% Settings to run the code
numEMGsForSync = 1; % Number of EMGs for synchronization
experimentNumber = 3; % Number of experiment to run {1, 2, 3, 4}

%% Adding the toolboxes required for the algorithm to work
addpath(genpath('Dataset and Code for Reading the Data and Evaluating the Results'));
addpath('Data Acquisition');
addpath('Pre-processing');
addpath('Feature Extraction');
addpath('Multiclass Neural Network Toolbox');
addpath('Classification');
addpath('Post-processing');
addpath('Rotation Toolbox');
rng('default'); % For reproducing the results

%% Defining the size of the subwindow and the stride for the sliding window
% approach
subWindowSize = 11; % Size of the subwindow
stride = 11; % Stride
%% Defining the architecture of the feed-forward ANN
inputSize = 68; % Size of the input layer
numNeuronsLayers = [inputSize, 500, 6]; % [inputLayerSize, hiddenLayerSize, outputLayerSize]
transferFunctions{1} = 'none'; % Activation functions of the input nodes
transferFunctions{2} = 'relu'; % Activation functions of the hidden neurons
transferFunctions{3} = 'softmax'; % Activation functions of the output neurons
options.reluThresh = 1e-4; % Threshold of the relu activation function
options.lambda = 1e-5; % Regularization factor for weight decay
options.numIterations = 500; % Number of epochs for training the neural network

switch experimentNumber
    %% Experiment 1
    case 1
        flagApplyRandomRotation_Train = false;
        flagCorrectTheRotation_Train = false;
        flagApplyRandomRotation_Test = false;
        flagCorrectTheRotation_Test = false;
    case 2
        %% Experiment 2
        flagApplyRandomRotation_Train = false;
        flagCorrectTheRotation_Train = false;
        flagApplyRandomRotation_Test = true;
        flagCorrectTheRotation_Test = false;
        
    case 3
        %% Experiment 3
        flagApplyRandomRotation_Train = false;
        flagCorrectTheRotation_Train = true;
        flagApplyRandomRotation_Test = true;
        flagCorrectTheRotation_Test = true;
        
    case 4
        %% Experiment 4
        flagApplyRandomRotation_Train = true;
        flagCorrectTheRotation_Train = true;
        flagApplyRandomRotation_Test = true;
        flagCorrectTheRotation_Test = true;
end

%% Initializing the variables
numTrainingEMGsPerGesturePerUser = 25;
startUser = 61;
endUser = 120;

%% Execution of the experiment
tStart = tic;
timesRecog = [];
timesClass = [];
historyIdxRotated = zeros(endUser - startUser + 1, 8);
historyIdxAntiRotated = zeros(endUser - startUser + 1, 8);
for userNum = startUser:endUser
    if userNum >= 61
        idx = userNum - 60;
    else
        idx = userNum;
    end
    fprintf('Processing user %d of %d\n', idx, endUser - startUser + 1);
    %% Training
    % Loading the EMGs for each testing user
    numUser = strcat('user', int2str(userNum));
    userData = getUser(numUser, 'Dataset and Code for Reading the Data and Evaluating the Results\data');
    % Extracting feature vectors and labels from each training EMG
    [Xtrain, Ytrain, idxRotated, idxAntiRotated, EMGsForSyncTest] =...
        getTrainingData(userData.training, ...
        numTrainingEMGsPerGesturePerUser,...
        numEMGsForSync,...
        flagApplyRandomRotation_Train,...
        flagCorrectTheRotation_Train,...
        subWindowSize, stride);
    historyIdxRotated(idx, :) = idxRotated;
    historyIdxAntiRotated(idx, :) = idxAntiRotated;
    % Balancing the training set so that each gesture has the same number of examples
    [Xtrain, Ytrain] = balanceTrainingData(Xtrain, Ytrain);
    % Normalization of the feature vectors, in the part of the covariances only
    meanCov = mean(Xtrain(:,1:28),2);
    stdCov = std(Xtrain(:,1:28),[],2);
    Xtrain = [((Xtrain(:,1:28) - meanCov)./stdCov), Xtrain(:, 29:68)];
    Xtrain = Xtrain( ~isnan(sum(Xtrain,2)), : );
    Ytrain = Ytrain( ~isnan(sum(Xtrain,2)) );
    softmaxNNModel = trainSoftmaxNN(Xtrain, Ytrain, numNeuronsLayers, transferFunctions, options);
    
    %% Testing
    % Computing the predictions of classification and recognition
    [predictions, timesRecog1, timesClass1] = ...
        classifyAndDetectGesture(userData.testing, softmaxNNModel, ...
        transferFunctions, options,...
        subWindowSize, stride,...
        flagApplyRandomRotation_Test,...
        flagCorrectTheRotation_Test,...
        EMGsForSyncTest, numEMGsForSync);
    timesRecog = [timesRecog; timesRecog1];
    timesClass = [timesClass; timesClass1];
    % Storing the results
    results.(genvarname(['user' int2str(userNum)])) = predictions;
    close all;
end
fprintf('Total time of testing: %3.2f s\n\n',toc(tStart));

%% Loading the mat files with the results
switch experimentNumber
    %% Experiment 1
    case 1
        load('experiment1.mat');
        %% Experiment 2
    case 2
        load('experiment2.mat');
        %% Experiment 3
    case 3
        if numEMGsForSync == 1 % Number of EMGs for synchronization
            load('experiment3_1SyncWO.mat');
        end
        if numEMGsForSync == 2 % Number of EMGs for synchronization
            load('experiment3_2SyncWO.mat');
        end
        if numEMGsForSync == 3 % Number of EMGs for synchronization
            load('experiment3_3SyncWO.mat');
        end
        if numEMGsForSync == 4 % Number of EMGs for synchronization
            load('experiment3_4SyncWO.mat');
        end
        %% Experiment 4
    case 4
        if numEMGsForSync == 1 % Number of EMGs for synchronization
            load('experiment4_1SyncWO.mat');
        end
        if numEMGsForSync == 2 % Number of EMGs for synchronization
            load('experiment4_2SyncWO.mat');
        end
        if numEMGsForSync == 3 % Number of EMGs for synchronization
            load('experiment4_3SyncWO.mat');
        end
        if numEMGsForSync == 4 % Number of EMGs for synchronization
            load('experiment4_4SyncWO.mat');
        end
end
%% Final results
% Computing the average of the times of classification
averageTimeOfClassification = mean(timesClass(:,end));
stdTimeOfClassification = std(timesClass(:,end));
fprintf('SUMMARY OF RESULTS\n\n');
fprintf('Average of the time of classification: %3.2f ms\n',averageTimeOfClassification*1000);
fprintf('Standard deviation of time of classification: %3.2f ms\n\n',stdTimeOfClassification*1000);
% Computing the average of the times of recognition
averageTimeOfRecognition = mean(timesRecog(:,end));
stdTimeOfRecognition = std(timesRecog(:,end));
fprintf('Average of the time of recognition: %3.2f ms\n',averageTimeOfRecognition*1000);
fprintf('Standard deviation of time of recognition: %3.2f ms\n\n',stdTimeOfRecognition*1000);
% Histogram of the times of classification
figure;
histogram( timesClass(:,end), 25);
title('Time of classification');
xlabel('Time [s]');
ylabel('Frequency');
% Histogram of the times of recognition
figure;
histogram( timesRecog(:,end), 25);
title('Time of recognition');
xlabel('Time [s]');
ylabel('Frequency');

% Plotting the confusion matrix
figure;
megaPlotConfusion(totalConfusionData);
% Analysis of the results of classification
userVector = fieldnames(classResult);
for kUser = 1:numel(userVector)
    nameUser = userVector{kUser};
    meanXUser(kUser) = mean(classResult.(nameUser));
end
figure;
bar(meanXUser);
hold on;
line([0, numel(meanXUser)], mean(meanXUser) * [1 1],'LineWidth', 2);
text(numel(meanXUser), mean(meanXUser), num2str(mean(meanXUser)),...
    'HorizontalAlignment', 'right', ...
    'fontSize', 20, 'FontAngle', 'italic', 'Color', 'r');
grid on;
title(['Average classification per user: ' num2str(mean(meanXUser))]);
xlabel('user');
ylabel('classification');
axis([0.5 numel(meanXUser) + 0.5 0 1]);
fprintf('Average of the classification accuracy: %3.2f %% \n', 100*mean(meanXUser));
fprintf('Standard deviation of the classification accuracy: %3.2f %% \n\n', 100*std(meanXUser));

% Analysis of the results of recognition
userVector = fieldnames(recognitionResult);
for kUser = 1:numel(userVector)
    nameUser = userVector{kUser};
    meanXUserRecog(kUser) = nanmean(recognitionResult.(nameUser));
end
figure;
bar(meanXUserRecog);
hold on;
line([0, numel(meanXUserRecog)], mean(meanXUserRecog) * [1, 1],'LineWidth', 2);
text(numel(meanXUserRecog), mean(meanXUserRecog), num2str(mean(meanXUserRecog)),...
    'HorizontalAlignment', 'right', ...
    'fontSize', 15, 'FontAngle', 'italic', 'Color', 'r');
grid on;
title('Average recognition per user');
xlabel('user');
ylabel('recognition');
axis([0.5 numel(meanXUser) + 0.5 0 max(meanXUserRecog)]);
fprintf('Average of the recognition accuracy: %3.2f %% \n', 100*mean(meanXUserRecog));
fprintf('Standard deviation of the recognition accuracy: %3.2f %% \n', 100*std(meanXUserRecog));