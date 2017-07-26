% computing distance between two vectors
% using Hamming metric 
% https://en.wikipedia.org/wiki/Hamming_distance
% arguments:
%	a - horizontal vector
%	b - horizontal vector
%
% returns:
%	the value of distance between input vectors

function f = hamming_distance(a, b)
	f = sum(a!=b);
end
