% Parameters:
%   X : input matrix; each input sample is a row vector
%   D : desired/target outputs (row-wise)
%   theta, dtheta: activation function and its derivative
%   maxiter: maximum number of iterations
%   epsilon: error threshold
%   lr: learning rate
%
% Output: weight cell array

function W = generic_backpropagation(X, D, layers, theta, dtheta, maxiter, epsilon, lr)

% initialize weights
W = {};
for i = 1:length(layers)-1
  %W{i} = randn(layers(i+1),layers(i));
  W{i} = randn(1,size(X,1));
end

for i = 1:maxiter
  GE = 0;

  for j = 1:size(X,3) 
    [Y, V] = forward_propagation(X(:,:,j), W, theta);
    [Delta, E] = local_gradient(D(j,:)', W, dtheta, Y, V);        
    W = update_weights(W, Delta, Y, lr);    
    GE = GE + E; 
  end
  
  fprintf('iter: %i, global error: %.4f\n', i, GE);
  
  if GE < epsilon, break; end
end

fprintf('Training finished. Global error is %.3f.\n', GE);