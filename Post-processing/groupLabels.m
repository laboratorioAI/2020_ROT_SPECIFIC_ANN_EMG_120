function YNew = groupLabels(Yhat,groupSize)
% This function takes as input a sequence of labels (i.e., vector), Yhat, and
% returns another sequence YNew, where length(YNew) <= length(Yhat). The
% elements of YNew are obtained from groups of consecuitive elements of Yhat
% by applying a mojority vote among the labels of each group
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec

numWindowObservations = length(Yhat);
YNew = [];
for groupNum = 1:groupSize:numWindowObservations
    startPoint = groupNum;
    endPoint = startPoint + groupSize - 1;
    if endPoint <= numWindowObservations
        YNew = [YNew, mode( Yhat(startPoint:endPoint) )];
    else
        YNew = [YNew, mode( Yhat(startPoint:end) )];
    end
end 
return