function [err,alpha,b] = train_test_svm_kernel(Ktrain, Ktest, Ytrain, Ytest, lambda)
m=size(Ktrain,1);

H=Ktrain.*(Ytrain*Ytrain')/(2*lambda);
f=-ones(1,m);

A=[];
b2=[];

Aeq=Ytrain';
beq=0;

LB=zeros(m,1);
UB=zeros(m,1)+1/m;

alpha=quadprog(H,f,A,b2,Aeq,beq,LB,UB);

idx=find(alpha>10^(-8) & alpha<1/m-10^(-8));
b=mean(Ytrain(idx)-Ktrain(idx,:)*(alpha.*Ytrain)/(2*lambda));

predictions=Ktest*alpha/(2*lambda)+b;
err=mean(sign(predictions)~=Ytest);

end

