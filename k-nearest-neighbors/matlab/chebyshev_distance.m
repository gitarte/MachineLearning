% computing distance between two vectors
% using Chebyshev formula 
% https://en.wikipedia.org/wiki/Chebyshev_distance
% arguments:
%	a - horizontal vector
%	b - horizontal vector
%
% returns:
%	the value of distance between input vectors

function f = chebyshev_distance(a, b)
	f = max(abs(a-b));
end
