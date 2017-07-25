function f = cosine_similarity(a, b)
	prod  = a*b';    % better equivalent of sum(a.*b)
	norm1 = norm(a); % better equivalent of sqrt(sum(a.^2));
	norm2 = norm(b); % better equivalent of sqrt(sum(b.^2));
	f = prod/(norm1*norm2);
end
