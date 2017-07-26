% computing distance between two vectors
% using Minkowski formula where p=1
% https://en.wikipedia.org/wiki/Euclidean_distance
% arguments:
%	a - horizontal vector
%	b - horizontal vector
%
% returns:
%	the value of distance between input vectors
% 
% By deffinition Manhattan distance can be comuted just as sum(abs(a-b))
% However I'll pay tribute to Hermann Minkowski whos distance formula
% is a generalisation of 
% Manhattan distance if p=1
% Euclidean distance if p=2, 
% Chebyshev distance if p reaches infinity at limit
function f = manhattan_distance(a, b)
	f = minkowski_distance(a, b, 1);
end
