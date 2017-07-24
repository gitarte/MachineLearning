function f = cosine_similarity(X, x)
	[m,n] = size(X);
	result = zeros(m,1);
	for i=1:m
		x1 = X(i,:);
		s = sum(x1.*x);
		p1 = sqrt(sum(x1.^2));
		p2 = sqrt(sum(x .^2));
		result(i,1) = s/(p1*p2);
	end
	f = result;
end
