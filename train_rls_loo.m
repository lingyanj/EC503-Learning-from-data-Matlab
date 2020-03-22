function [w,b,err_tr,err_loo]=train_rls_loo(X,y,lambda)

[m,d]=size(X);

tildeX=[ones(m,1) X];
tildeI=lambda*eye(d+1);
tildeI(1,1)=0;
Cinv=pinv(tildeX'*tildeX+tildeI);
tildew=Cinv*tildeX'*y;
b=tildew(1);
w=tildew(2:end);

for i=1:m
    err_loo(i,1)=(tildeX(i,:)*tildew-y(i))./(1-tildeX(i,:)*Cinv*tildeX(i,:)');
end
err_tr=mean((tildeX*tildew-y).^2);
err_loo=mean(err_loo.^2);