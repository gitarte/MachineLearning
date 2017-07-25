% computing cosine of an angle between two vectors
% using the Euclidean dot product formula 
% https://en.wikipedia.org/wiki/Cosine_similarity
% arguments:
%	a - horizontal vector
%	b - horizontal vector
%
% returns:
%	the value of cosine between input vectors

function f = cosine_similarity(a, b)
	prod  = a*b';    % better equivalent of sum(a.*b)
	norm1 = norm(a); % better equivalent of sqrt(sum(a.^2));
	norm2 = norm(b); % better equivalent of sqrt(sum(b.^2));
	f = prod/(norm1*norm2);
end
