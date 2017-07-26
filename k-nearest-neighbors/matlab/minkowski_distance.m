% computing distance between two vectors
% using Minkowski formula 
% https://en.wikipedia.org/wiki/Minkowski_distance
% arguments:
%	a - horizontal vector
%	b - horizontal vector
%	p - skalar valu representing the exponent value
%       for p=1 the Minkowski distance is an equivalent of Manhattan distance
%       for p=2 the Minkowski distance is an equivalent of Euclidean distance
%       for p reaching infinity at limit Minkowski distance is an equivalent of Chebyshev distance
% returns:
%	the value of distance between input vectors

function f = minkowski_distance(a, b, p)
	f = sum(abs(a-b).^p)^(1/p);
end
