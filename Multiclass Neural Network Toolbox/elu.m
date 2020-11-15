function y = elu(x)
% elu(x) computes the value of the elu transfer function for the elements
% of x
y = (x > 0).*x + (x <= 0).*(exp(x) - 1);
return