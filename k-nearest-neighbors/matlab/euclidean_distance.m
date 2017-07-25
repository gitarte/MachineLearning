% computing distance between two vectors
% using Euclidean distance 
% https://en.wikipedia.org/wiki/Euclidean_distance
% arguments:
%	a - horizontal vector
%	b - horizontal vector
%
% returns:
%	the value of cosine between input vectors

function f = euclidean_distance(a, b)
	f = sqrt(sum((a-b).^2));
end
