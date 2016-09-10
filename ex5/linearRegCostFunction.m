function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

one_over_2m = 1 / (2*m);
%size(X)  % 12    2
%size(theta) % 2   1
%size(y)  % 12    2

H = X * theta;
J = one_over_2m * sum((H - y).^2);


penalty = lambda * one_over_2m * (sum([0; theta(2:end)].^2));

J = J + penalty;

penalty_grad = (lambda/m) * theta(2:end);


grad0 = (1/m) * sum((H - y).*X(:,1));
grad1 = (1/m) * sum((H - y).*X(:,2:end),1)' + penalty_grad;
grad = [grad0; grad1];






% =========================================================================

grad = grad(:);

end
