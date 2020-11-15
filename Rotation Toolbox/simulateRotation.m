function idxRotated = simulateRotation()
% This function rotates randomly, with a uniform distribution in the set
% {-3, -2, -1, 0, +1, +2, +3, +4}, the channels of an 8-channel EMG 
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec

numChannels = 8;
% Angle of rotation
randAngle = 4 - round( numChannels*rand );
randAngle(randAngle == -4) = 4; %Position -4 is the same as position +4

% Rotation of the channels
idx = 1:numChannels;
idxRotated = mod(idx + randAngle, numChannels);
idxRotated(idxRotated == 0) = numChannels;
return