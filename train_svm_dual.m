function [w,b] = train_svm_dual(X,y,lambda)
[m,d]=size(X);

H=(X*X').*(y*y')/(2*lambda);
f=-ones(1,m);

A=[];
b2=[];

Aeq=y';
beq=0;

LB=zeros(m,1);
UB=zeros(m,1)+1/m;

alpha=quadprog(H,f,A,b2,Aeq,beq,LB,UB);

w=zeros(d,1);
for i=1:m
    w=w+alpha(i)*y(i)*X(i,:)'/(2*lambda);
end
idx=find(alpha>0.00001 & alpha<1/m-0.00001);
b=mean(y(idx)-X(idx,:)*w);
end

