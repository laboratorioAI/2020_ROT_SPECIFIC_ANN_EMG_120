function idxAntiRotated = correctTheRotation(EMGsForSync)
% This function implements the algorithm for correction of the orientation
% by computing the channel of maximum energy of the set EMGsForSync of EMGs
% of synchronization. The vector idxAntiRotated contains the sequence in
% which the channels of the EMG should be rearranged
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec

numEMGsForSync = length(EMGsForSync);
% Rectification and filtering the EMGs for synchronization
% Low-pass Butterworth filter design
fc = 10; %Cutoff frequency
fs = 200; %Sampling frequency
[b,a] = butter(2,fc/(fs/2));
numChannels = size(EMGsForSync{1},2);
energyOfEachChannel = zeros( 1, numChannels );
for EMGForSyncNum = 1:numEMGsForSync
    % Filtering the EMG
    EMGsForSyncEnvelope = filter(b,a,abs(EMGsForSync{EMGForSyncNum}));
    % Computing the energy of each channel
    energyVal = trapz( EMGsForSyncEnvelope.^2 )/size(EMGsForSync{EMGForSyncNum},1);
    energyOfEachChannel = energyOfEachChannel + energyVal;
end
% Finding the channel with maximum energy
[dummy, idx] = max(energyOfEachChannel);
idxAntiRotated = mod( idx:(idx+7), numChannels );
idxAntiRotated(idxAntiRotated == 0) = numChannels;
return