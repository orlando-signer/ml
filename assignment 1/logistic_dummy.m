function [ theta ] = logistic_dummy( data, labels )

theta = [ randn( size(data,2), 1 ); 0 ];

end

