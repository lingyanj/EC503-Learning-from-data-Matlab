function [w,b] = incremental_train_rls(X,y,lambda)
[m,d]=size(X);
Xtilde=[ones(m,1) X];

% Here, we have to construct the solution incrementally starting with the
% solution with no training data.
% To make things nice, change tildeI to [eps/lambda 0^T; 0 I], for epsilon
% arbiratry small.
% When eps goes to 0, we recover the correct solution, but for any eps>0,
% this is diagonale and invertible! Note that this corresponds to add
% to the objective function the term eps*b^2.
%
% Note that in Matlab 'eps' is the smallest number we can represent, so
% numerically this is basically equivalent to put there a 0.

% first inverted C with no samples
inverseC=[1/eps zeros(1,d); zeros(d,1) eye(d)/lambda];
for i=1:m
    % let's add sample i to C and calculate the new inverse with
    % Sherman-Morrison
    q=inverseC*Xtilde(i,:)';
    inverseC = inverseC - q*q'/(1+Xtilde(i,:)*q);
end

wtilde=inverseC*Xtilde'*y;
b=wtilde(1);
w=wtilde(2:end);

end
