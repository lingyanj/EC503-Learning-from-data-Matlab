clear
close all

% create random training data
% Any random data is fine! Here, we use something for which we know what is
% the correct solution, to verify that RLS works too
d=10;
m=1000;
% generate the random correct solution
u=randn(d,1);
b=randn;
% generate the training points
X=randn(m,d);
% generate the labels as the one given by the vector u plus gaussian noise
y=X*u+b+.1*randn(m,1);


% train RLS
lambda=0.1;
[w,b]=train_rls(X,y,lambda);


% How well did we do?
% (this was not a required plot, yet it is nice to see that it works)
figure(1)
plot(w)
hold on
plot(u,'k')


% train RLS with the incremental method
[w2,b2]=incremental_train_rls(X,y,lambda);


% How well did we do?
% (plotting or verifying numerically is fine)
figure(2)
plot(w)
hold on
plot(w2,'k')
