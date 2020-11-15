function Y = postProcess(Y)
% This function returns the current label Y(i) based on the past Y(i-1) and
% future Y(i+1) labels.
%
% Marco E. Benalcázar, Ph.D.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.benalcazar@epn.edu.ec

l = length(Y);
Y(1) = 6;
for i = 2:(l-1)
    if (Y(i-1) == Y(i+1)) && (Y(i) ~= Y(i-1))
            Y(i)= Y(i-1);
    end
   if (Y(i-1) ~= Y(i+1)) && (Y(i) ~= Y(i-1)) && (Y(i) ~= Y(i+1))
            Y(i) = Y(i-1);
   end
end
Y(l) = 6;
return