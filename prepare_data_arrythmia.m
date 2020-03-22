clear 
 
load arrhythmia.mat

[m,d]=size(X);

med = median(X,'omitnan');
X = fillmissing(X,'constant',med);


% for numerical stability, makes all the features with zero mean and std 1
% Also, this gives the same importance to all the features
X=X-repmat(mean(X),m,1);
X=X./(eps+repmat(std(X),m,1));

% random train/test split
idx=randperm(m);
X=X(idx,:);
Y=double(Y(idx));
Y(Y==0)=-1; % convert labels
m_train=round(0.8*m);
Xtrain = X(1:m_train,:);
Xtest = X(m_train+1:end,:);
Ytrain = Y(1:m_train,:);
Ytest = Y(m_train+1:end,:);

save arrhythmia_training_test.mat