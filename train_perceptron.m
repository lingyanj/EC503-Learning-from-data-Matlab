function [w,b,average_w,average_b]=train_perceptron(X,y)

[m,d]=size(X);
tildeX=[ones(m,1) X];

tildew=zeros(d+1,1);
average_tildew=zeros(d+1,1);
for i=1:m
    if y(i)*(tildeX(i,:)*tildew)<=0
        tildew=tildew+y(i)*tildeX(i,:)';
    end
    average_tildew=average_tildew+tildew;
end
average_tildew=average_tildew/m;
b=tildew(1);
w=tildew(2:end);
average_b=average_tildew(1);
average_w=average_tildew(2:end);

