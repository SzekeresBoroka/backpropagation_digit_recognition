%Parameters:
%   x - column vector with input
%   W - cell array with weight matrices
%   theta - activation function
%
%Returns:
%   Y - cell array with outputs
%   V - cell array with total (weighted) inputs

function [Y, V] = forward_propagation(x, W, theta)

V = {x};
Y = {x};

for i = 1:length(W)
  V = {V{:} W{i}.*Y{end}};
  Y = {Y{:} theta(V{end})};
end