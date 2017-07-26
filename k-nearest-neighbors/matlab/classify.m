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

% Whenever a similarity is computed according to some kind of distance
% the computed value is subtracted from 1. This fulfills Frey and Dueck 
% definition of similarity, but in such fashion that maximum of
% similarity is equal to 1, because the minimal distance is equal to 0.
% 
% similarity = 1 - distance

% USING MANHATTAN DISTANCE
for i=1:mTest
	xi = Xtest(i,:);
	for j=1:m
		xj = X(j,:);
		similarity(j,1) = 1 - manhattan_distance(xi, xj);
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
printf('euclidean_distance: %f\n',eff);

% USING EUCLIDEAN DISTANCE
for i=1:mTest
	xi = Xtest(i,:);
	for j=1:m
		xj = X(j,:);
		
		similarity(j,1) = 1 - euclidean_distance(xi, xj);
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
printf('euclidean_distance: %f\n',eff);

% USING CHEBYSHEV DISTANCE
for i=1:mTest
	xi = Xtest(i,:);
	for j=1:m
		xj = X(j,:);
		similarity(j,1) = 1 - chebyshev_distance(xi, xj);
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
printf('chebyshev_distance: %f\n',eff);

% USING HAMMING DISTANCE
for i=1:mTest
	xi = Xtest(i,:);
	for j=1:m
		xj = X(j,:);
		similarity(j,1) = 1 - hamming_distance(xi, xj);
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
printf('hamming_distance:   %f\n',eff);

% USING COSINE SIMILARITY
% In case of cosine similarity I do not subtract it from 1, because 
% cosine has values in range <-1; 1> where 1 already means 
% "the most similar"
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
printf('cosine_similarity:  %f\n',eff);
