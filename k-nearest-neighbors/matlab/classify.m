clear all;
close all,
clc;

data = load('../../datasets/hash-recogintion/dataset.csv');
data = [ones(4000,1) data];

X = data(1:2800, 1:9);
y = data(1:2800, 10);

Xtest = data(2801:end, 1:9);
ytest = data(2801:end, 10);

k = 3;
[m, n]         = size(X);
[mTest, nTest] = size(Xtest);

prediction = zeros(mTest,1);
similarity = zeros(m,    1);

for i=1:mTest
	xi = Xtest(i,:);
	for j=1:m
		xj = X(j,:);
		similarity(j,1) = euclidean_distance(xi, xj);
	end
	[_, idx] = sort(similarity,'ascend');
	idx = idx(1:k,:)';
	a   = sum(y(idx,1));
	b   = k-a;
	if (a>=b)
		prediction(i)=1;
	else
		prediction(i)=0;
	end	
end
eff = sum((prediction==ytest))/mTest*100;
printf('euclidean_distance: %f\n',eff);

for i=1:mTest
	xi = Xtest(i,:);
	for j=1:m
		xj = X(j,:);
		similarity(j,1) = cosine_similarity(xi, xj);
	end
	[_, idx] = sort(similarity,'descend');
	idx = idx(1:k,:)';
	a   = sum(y(idx,1));
	b   = k-a;
	if (a>=b)
		prediction(i)=1;
	else
		prediction(i)=0;
	end	
end
eff = sum((prediction==ytest))/mTest*100;
printf('cosine_similarity: %f\n',eff);
