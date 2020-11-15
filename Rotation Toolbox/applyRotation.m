function EMGout = applyRotation(EMGin, idx)
% This function rotates the channels (i.e., columns) of the EMGin
% accoriding to the sequence of indexes idx
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec
EMGout = EMGin(:, idx);
return