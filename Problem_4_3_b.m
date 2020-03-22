load arrhythmia_training_test.mat

folds=3;

lambdas=10.^(-12:8);

% Linear
err_linear=zeros(1,numel(lambdas));
for i=1:numel(lambdas)
    err_linear(i) = cross_validation(Xtrain, Ytrain, folds, 1, 0, lambdas(i));
end

% Polynomial
err_poly=zeros(numel(lambdas),2);
for degree=2:3
    for i=1:numel(lambdas)
        err_poly(i,degree) = cross_validation(Xtrain, Ytrain, folds, 2, degree, lambdas(i));
    end
end

%Gaussian
gammas=10.^(-10:5);
err_gauss=zeros(numel(lambdas),numel(gammas));
for j=1:numel(gammas)
    for i=1:numel(lambdas)
        err_gauss(i,j) = cross_validation(Xtrain, Ytrain, folds, 3, gammas(j), lambdas(i));
    end
end

fprintf('Best cross-validation err linear kernel: %f\n', min(err_linear));
fprintf('Best cross-validation err poly kernel (degree 2): %f\n', min(err_poly(:,2)));
fprintf('Best cross-validation err poly kernel (degree 3): %f\n', min(err_poly(:,3)));
fprintf('Best cross-validation err Gaussian kernel: %f\n', min(err_gauss(:)));

%Linear kernel
%Retrain on everything with optimal lambda and test on test set
[~,idx_min]=min(err_linear);
Ktrain=compute_kernel_matrix(Xtrain,Xtrain,1,0);
Ktest=compute_kernel_matrix(Xtest,Xtrain,1,0);
test_err=train_test_svm_kernel(Ktrain, Ktest, Ytrain, Ytest, lambdas(idx_min));
fprintf('Test err tuned linear kernel: %f\n', test_err);

%Poly kernel, degree 2
%Retrain on everything with optimal lambda and test on test set
[~,idx_min]=min(err_poly(:,2));
Ktrain=compute_kernel_matrix(Xtrain,Xtrain,2,2);
Ktest=compute_kernel_matrix(Xtest,Xtrain,2,2);
test_err=train_test_svm_kernel(Ktrain, Ktest, Ytrain, Ytest, lambdas(idx_min));
fprintf('Test err tuned poly kernel (degree 2): %f\n', test_err);

%Poly kernel, degree 3
%Retrain on everything with optimal lambda and test on test set
[~,idx_min]=min(err_poly(:,3));
Ktrain=compute_kernel_matrix(Xtrain,Xtrain,2,3);
Ktest=compute_kernel_matrix(Xtest,Xtrain,2,3);
test_err=train_test_svm_kernel(Ktrain, Ktest, Ytrain, Ytest, lambdas(idx_min));
fprintf('Test err tuned poly kernel (degree 3): %f\n', test_err);

%Gauss kernel
%Retrain on everything with optimal lambda and test on test set
[value, index] = min(err_gauss(:));
[row, col] = ind2sub(size(err_gauss), index);
Ktrain=compute_kernel_matrix(Xtrain,Xtrain,3,gammas(col));
Ktest=compute_kernel_matrix(Xtest,Xtrain,3,gammas(col));
test_err=train_test_svm_kernel(Ktrain, Ktest, Ytrain, Ytest, lambdas(row));
fprintf('Test err tuned Gaussian kernel: %f\n', test_err);
