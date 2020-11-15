function [predictions, timesRecog, timesClass] = ...
    classifyAndDetectGesture(testingData, softmaxNNModel,...
    transferFunctions, options,...
    subWindowSize, stride,...
    flagApplyRandomRotation,...
    flagCorrectTheRotation,...
    EMGsForSyncTest, numEMGsForSync)
% This function classifies and recognizes a gesture given an EMG and
% trained ANN
%
% Inputs:
% testingData:   Cell of Nx1 elements, where the element testingData{i}
%                contains the EMG, the recognition returned by the Myo
%                Armband (poseMyo) and the rotation matrices (rot) for the ith
%                testing sample recorded for a given user from set of N users. 
%                It is important to remember that the sampling frequency of
%                the EMG is 200Hz and the sampling frequency of the orientation is 50Hz
%
% softmaxNNModel: Cell of 1x(L-1) elements, where the element
%                 softmaxNNModel(i) contains the weights, of a trained
%                 multiclass feed-forward neural newtwork, that conect the layer i 
%                 with the layer i + 1. In this case, the network has a
%                 total of L layers
%
% transferFunctions: Cell of 1xL elements, where the element
%                 transferFunctions(i) contains the activation functions for
%                 the units of the ith layer of the multiclass feed-forward 
%                 neural newtwork. The activation functions of the input
%                 and output units must always be 'none' and 'softmax',
%                 respectively
%
% options:       Structure that contains the threshold for the ReLu
%                activation function (reluThresh), the weight for the
%                weight-decay and the number of epochs (numIterations) 
%                for training the neural network
% 
% idxAntiRotated: Vector of 1x8 positive integers that correspond to a
%                 permutation of the elements of the set {1, 2, ..., 8}.
%                 The first element of this vector is the number of the
%                 channel of the EMG that has the highest power for the
%                 gesture waveOut. If, for example, the first element of
%                 this vector is 4, the next 7 elements are 5, 6, 7, 8, 1,
%                 2, 3
%
% subWindowSize: Positive integer that indicates the size of the subwindow
%
% stride:        Positive integer that indicates the separation between two
%                consecutive subwindow observations
% Outputs:
% predictions:   Cell of Nx1 elements, where the element predictions{i}
%                contains the results of classification (classPrediction)
%                and the results of recognition (predictionsMatrix). The
%                results of recognition contain a vector of labels (vectorOfLabels)
%                and a vector of times (vectorOfTimes), where
%                vectorOfTimes(j) is the time where the label
%                vectorOfLabels(j) was predicted
% 
% timesRecog:   Vector of Nx6 elements, where timesRecog(i,1) is the time
%               of pre-processing and feature extraction, timesRecog(i,2)
%               is the time of normalizing the training examples, timesRecog(i,3)
%               is the time of classification, timesRecog(i,4) is the
%               time of computing the equivalent class of the predictions
%               for a set of subwindows, timesRecog(i,5) is the time of
%               pos-processing, and timesRecog(i,6) is the sum of all the 5
%               times described above. The index i denotes the number of
%               EMG being processed
% 
% timesClass;   Vector of Nx5 elements, where timesRecog(i,1) is the time
%               of pre-processing and feature extraction, timesRecog(i,2)
%               is the time of normalizing the training examples, timesRecog(i,3)
%               is the time of classification, timesRecog(i,4) is the
%               time of computing the equivalent class for the ith EMG, and
%               timesRecog(i,5) is the sum of all the 4 times described above
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec

numEMGs = length(testingData); % Number of EMGs of test
% Initialization of variables
timesRecog = zeros(numEMGs,6);
timesClass = zeros(numEMGs,5);
windowSettings.windowSize = subWindowSize;
windowSettings.stride = stride;
predictions = cell(numEMGs,1);
% Random rotation
if flagApplyRandomRotation
    % Simulating the rotation of the bracelet
    idxRotated = simulateRotation();
else
    idxRotated = 1:size(testingData{1}.emg,2);
end
if flagCorrectTheRotation
    for i = 1:numEMGsForSync
        EMGsForSyncTest{i} = applyRotation(EMGsForSyncTest{i}, idxRotated);
    end
    % Correcting the rotation of the bracelet
    idxAntiRotated = correctTheRotation(EMGsForSyncTest);
else
    idxAntiRotated = idxRotated;
end

for EMGNum = 1:numEMGs 
    % Reading the test EMG
    testEMG = testingData{EMGNum}.emg;
    % Simulating the rotation of the bracelet
    testEMGrotated = applyRotation(testEMG, idxRotated);
    % Correction of rotation
    tic
    if flagCorrectTheRotation
        testEMGantirotated = applyRotation(testEMGrotated, idxAntiRotated);
    else
        testEMGantirotated = testEMGrotated;
    end
    % Feature extraction
    Xtest = getFeatureVectors(testEMGantirotated, [], [], windowSettings);
    numWindowObservations  = size(Xtest,1);
    timesRecog(EMGNum,1) = (toc/numWindowObservations)*18;
    timesClass(EMGNum,1) = toc;
    
    % Normalizing the feature vectors
    tic;
    meanCov = mean(Xtest(:,1:28),2);
    stdCov = std(Xtest(:,1:28),[],2);
    Xtest = [((Xtest(:,1:28) - meanCov)./stdCov), Xtest(:, 29:68)];
    timesRecog(EMGNum,2) = (toc/numWindowObservations)*18;
    timesClass(EMGNum,2) = toc;
    
    % Classification of the feature vectors
    tic;
    [dummy, A] = forwardPropagation(Xtest, softmaxNNModel, transferFunctions, options);
    P = A{end};
    [maxP, Yhat] = max(P, [], 2);
    Yhat(maxP < 0.45) = 6;
    timesRecog(EMGNum,3) = (toc/numWindowObservations)*18;
    timesClass(EMGNum,3) = toc;

    % Classification of the EMG
    tic;
    YHatWithNoRest = Yhat(Yhat ~= 6);
    EMGLabel = mode(YHatWithNoRest);
    if isnan( EMGLabel )
        EMGLabel = 6;
    end
    predictions{EMGNum}.classPrediction = EMGLabel;
    timesClass(EMGNum,4) = toc;
    timesClass(EMGNum,5) = sum(timesClass(EMGNum,:));
    
    % Recognition: defining the label of each group of 6 consecutive window observations
    % Vector of labels
    groupSize = 6;
    tic;
    vectorOfLabels = groupLabels(Yhat, groupSize);
    timesRecog(EMGNum,4) = (toc/(numWindowObservations/6))*3;
    tic;
    vectorOfLabels = postProcess(vectorOfLabels);
    timesRecog(EMGNum,5) = toc/(numWindowObservations-2);
    % Vector of times
    numLabels = length(vectorOfLabels);
    vectorOfTimes = zeros(1,numLabels);
    for i=1:numLabels
        vectorOfTimes(i) = ((1000/numLabels)/200)*i-(((1000/numLabels)/200))/2;
    end
    timesRecog(EMGNum,6) = sum(timesRecog(EMGNum,:));
    predictions{EMGNum}.predictionsMatrix = [vectorOfLabels;vectorOfTimes];   
end
return