function [vectorX, vectorY, idxRotated, idxAntiRotated, EMGsForSyncTest] = ...
    getTrainingData(trainingData, numTrainingEMGsPerGesture, numEMGsForSync,...
    flagApplyRandomRotation,...
    flagCorrectTheRotation,...
    windowSize, stride)
% This function extracts feature vectors from a set of training EMGs using
% a slidding window approach
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec

options.windowSize = windowSize;
options.stride = stride;
vectorX = [];
vectorY = [];
%% Random rotation of the bracelet for training
if flagApplyRandomRotation
    % Simulating the rotation of the bracelet
    idxRotated = simulateRotation();
else
    idxRotated = 1:size(trainingData{1}.emg,2);
end
%% Correction of the orientation for training
if flagCorrectTheRotation
    % Extracting the EMG for synchronization (we use wave-out)
    idxForSyncGestures = 31:(31+(numTrainingEMGsPerGesture - 1));
    % Selecting randomly one sample of waveOut to correct rotation
    [dummy, idx] = sort( rand(1, numTrainingEMGsPerGesture) );
    EMGsForSyncTrain = cell(numEMGsForSync, 1);
    for i = 1:numEMGsForSync
        EMGsForSyncTrain{i} = applyRotation(trainingData{idxForSyncGestures(idx(i))}.emg, idxRotated);
    end
    % Correcting the rotation of the bracelet
    idxAntiRotated = correctTheRotation(EMGsForSyncTrain);
else    
    idxAntiRotated = idxRotated;
end
%% EMGs for correction of the orientation for testing
% Extracting the EMG for synchronization (we use wave-out)
idxForSyncGestures = 31:(31+(numTrainingEMGsPerGesture - 1));
% Selecting randomly one sample of waveOut to correct rotation
[dummy, idx] = sort( rand(1, numTrainingEMGsPerGesture) );
EMGsForSyncTest = cell(numEMGsForSync, 1);
for i = 1:numEMGsForSync
    EMGsForSyncTest{i} = trainingData{idxForSyncGestures(idx(i))}.emg;
end

%% Getting the training data
for i = 6:(numTrainingEMGsPerGesture+5) % We only extract data from the EMGs of each gesture and exclude the EMGs of the class relax
    % getting the EMGs for each class
    EMG_A = trainingData{i}.emg;
    EMG_B = trainingData{i+25}.emg;
    EMG_C = trainingData{i+50}.emg;
    EMG_D = trainingData{i+75}.emg;
    EMG_E = trainingData{i+100}.emg;
    
    % Applying the rotation to the EMGs
    EMG_Arotated = applyRotation(EMG_A, idxRotated);
    EMG_Brotated = applyRotation(EMG_B, idxRotated);
    EMG_Crotated = applyRotation(EMG_C, idxRotated);
    EMG_Drotated = applyRotation(EMG_D, idxRotated);
    EMG_Erotated = applyRotation(EMG_E, idxRotated);
    
    % Applying the antirotation to the EMGs
    EMG_Acorrected = applyRotation(EMG_Arotated, idxAntiRotated);
    EMG_Bcorrected = applyRotation(EMG_Brotated, idxAntiRotated);
    EMG_Ccorrected = applyRotation(EMG_Crotated, idxAntiRotated);
    EMG_Dcorrected = applyRotation(EMG_Drotated, idxAntiRotated);
    EMG_Ecorrected = applyRotation(EMG_Erotated, idxAntiRotated);
    
    % Getting feature vectors and labels for each EMG    
    [featureVectors_A, labels_A] = getFeatureVectors(EMG_Acorrected, trainingData{i}.emgGroundTruth, 1, options);
    [featureVectors_B, labels_B] = getFeatureVectors(EMG_Bcorrected, trainingData{i+25}.emgGroundTruth, 2, options);
    [featureVectors_C, labels_C] = getFeatureVectors(EMG_Ccorrected, trainingData{i+50}.emgGroundTruth, 3, options);
    [featureVectors_D, labels_D] = getFeatureVectors(EMG_Dcorrected, trainingData{i+75}.emgGroundTruth, 4, options);
    [featureVectors_E, labels_E] = getFeatureVectors(EMG_Ecorrected, trainingData{i+100}.emgGroundTruth, 5, options);
    vectorX = [vectorX; featureVectors_A; featureVectors_B; featureVectors_C; featureVectors_D; featureVectors_E];
    vectorY = [vectorY; labels_A; labels_B; labels_C; labels_D; labels_E];
end
vectorY(vectorY == 0) = 6; % Remapping the relax from class 0 to class 6
return