function [w,b] = train_svm_primal(X,y,lambda)
[m,d]=size(X);

H=zeros(d+m+1,d+m+1);
for i=1:d
    H(1+i,1+i)=2*lambda;
end

f=zeros(1,d+m+1);
f(d+1+1:end)=1/m;
tildeX=[ones(m,1) X];
A=[tildeX.*repmat(-y,[1,d+1]) -eye(m)];
A=[A;zeros(m,d+1) -eye(m)];
b2=zeros(2*m,1);
b2(1:m)=-1;
sol=quadprog(H,f,A,b2);

b=sol(1);
w=sol(2:d+1);

end

