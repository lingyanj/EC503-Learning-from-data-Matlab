clear all;
close all;
addpath /home/bremen/Projects/libsvm-3.22/matlab/

x=randn(5000,2)+repmat([.5 , .5],[5000,1]);
x=[x; 2*randn(5000,2)+repmat([3 , 10],[5000,1])];
plot(x(1:5000,1), x(1:5000,2),'x')
hold on
plot(x(5001:end,1), x(5001:end,2),'o')
y=ones(10000,1);
y(5001:end)=-1;

model=svmtrain(y,x,'-t 0 -c 100');
idx=model.sv_indices(find(abs(model.sv_coef)>99.9));
x(idx,:)=[];
y(idx)=[];

figure(2)
plot(x(1:5000,1), x(1:5000,2),'x')
hold on
plot(x(5001:end,1), x(5001:end,2),'o')

idx=randperm(size(x,1));
x=x(idx,:);
y=y(idx);

Xtrain=x(1:7000,:);
ytrain=y(1:7000);
Xtest=x(7001:end,:);
ytest=y(7001:end);

%save -v4 synth_data.mat Xtrain ytrain Xtest ytest

model=svmtrain(ytrain,Xtrain,'-t 0 -c 0.01');
[tmp, acc001, tmp2]=svmpredict(ytest,Xtest,model);
w_svm=zeros(1,2);
for i=1:numel(model.sv_indices)
  w_svm=w_svm+model.sv_coef(i)*Xtrain(model.sv_indices(i),:);
end
z1_svm=1/w_svm(1); z2_svm=-1/w_svm(2);
b_svm=-model.rho;
a=min(Xtrain(:,1)):0.01:max(Xtrain(:,1));
h1=plot(a,z2_svm/z1_svm*a-b_svm/w_svm(2),'r');

model=svmtrain(ytrain,Xtrain,'-t 0 -c 1000');
[tmp, acc100, tmp2]=svmpredict(ytest,Xtest,model);

w_svm=zeros(1,2);
for i=1:numel(model.sv_indices)
  w_svm=w_svm+model.sv_coef(i)*Xtrain(model.sv_indices(i),:);
end
z1_svm=1/w_svm(1); z2_svm=-1/w_svm(2);
b_svm=-model.rho;
h2=plot(a,z2_svm/z1_svm*a-b_svm/w_svm(2),'g');


[w,b]=train_rls(Xtrain,ytrain,1);
numel(find(sign(Xtest*w+b)==ytest))
z1=1/w(1); z2=-1/w(2);
a=min(x(:,1)):0.01:max(x(:,1));
h3=plot(a,z2/z1*a-b/w(2),'k');
hold on;

legend([h1 h2 h3], 'SVM C=0.01', 'SVM C=1000', 'RLS, \lambda=1')

fprintf('Err svm lambda=0.01: %f\n', acc001(1));
fprintf('Err svm lambda=100: %f\n', acc100(1));