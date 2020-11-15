function featureVector = observationToFeatureVector(A)
% This function maps an EMG observation (i.e., EMG segment) into a feature
% vector
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec

% Filtering and rectification of the EMG
Aplus = filterAndRectify(A);
% Covarince between the channels of the EMG
covMatrix = cov(Aplus,1);
idx = (1:8).' < (1:8);
covVector = covMatrix(idx)';
% Band power
bandPowerFeat = bandpower(A);
% Mean frequency
meanFrequency = meanfreq(A);
% Occupied bandwidth:
occupiedBandWidth = obw(A);
% Mean Absolute Value
MAV = sum(abs(A))/size(A,1);
% Wavefrom length 
WL = sum(abs(diff(A)));
% Definition of the feature vector
featureVector = [covVector, bandPowerFeat, meanFrequency, occupiedBandWidth, MAV, WL];
return