function [phi, mu0, mu1, Sigma] = gda_dummy( data, labels )

phi = 0.5;
mu0 = rand(size(data,1),1);
mu1 = rand(size(data,1),1);
Sigma = eye(size(data,1));

end

