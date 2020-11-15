function EMGOutput = filterAndRectify(EMGInput)
% This function extracts the envelope of an EMG by first reactifying the
% EMG and the applying a low-pass filter. The envelope of an EMG is assumed
% to cacry the information of the gesture performed by a user
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec

% Low-pass Butterworth filter design
fc = 1; %Cutoff frequency 
fs = 200; %Sampling frequency
[b,a] = butter(2,fc/(fs/2));
% Filtering the EMG
EMGOutput = filter(b,a,abs(EMGInput));
return