function theta = logistic_sgd_orlando_signer ( data , labels )
% logistic regression using stochastic gradient descent
% input : data : dxn matrix , where d is the dimensionality and
%       n is the number of training samples . The column
%       data (: , i ) represents the training sample x ˆ i
%       labels : 1xn row vector , labels ( i ) contains y ˆ i
% output : theta : dx1 column vector , the parameters of the
%       classifier

alpha = 0.1;
n = 200; % iterations
N = size(data,1);
theta = zeros(2,n);
theta(:,1) = [0;0]; % init
for i=2:n
    idx = floor(N*rand(1,1)) + 1; % mod(i,N) + 1;
    disp(labels(idx));
    disp(theta(:,i-1));
    disp(data(:,idx)');
    err = labels(idx)-theta(:,i-1)'*data(:,idx);
    theta(:,i) = theta(:,i-1) + alpha * err * data(:,idx);
end
thetaEnd = theta(:,n);


end