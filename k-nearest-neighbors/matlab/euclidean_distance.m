function f = euclidean_distance(X, x)
	d = bsxfun(@minus,X,x); 
	p = d.^2;
	s = sum(p');
	f = sqrt(s)';
end
