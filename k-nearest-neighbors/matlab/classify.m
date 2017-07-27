clear all;
close all,
clc;

% CONFIGURATION
k = 3;		% number of neighbors that will vote
r = 0.9; 	% the size of teaching set

% LOADING DATASET
data = load('../../datasets/hash-recogintion/dataset.csv');
[mData, _] = size(data);
data = [ones(mData,1) data];

% SPLIT DATASET INTO TEACHING AND TESTING BATCH
lastTeachingIdx = ceil(0.7*mData);
firstTestingIdx = lastTeachingIdx + 1;
X     = data(1:lastTeachingIdx,   1:9); % teaching examples
y     = data(1:lastTeachingIdx,   10);  % teaching classes
Xtest = data(firstTestingIdx:end, 1:9); % testing  examples
ytest = data(firstTestingIdx:end, 10);  % testing  classes


[m, _]     = size(X);     % of course m = lastTeachingIdx but this is true only in Matlab
[mTest, _] = size(Xtest);

similarity = zeros(m,    1);
prediction = zeros(mTest,1);

% Whenever a similarity is computed according to some kind of distance
% the computed value is subtracted from 1. This fulfills Frey and Dueck 
% definition of similarity, but in such fashion that maximum of
% similarity is equal to 1, because the minimal distance is equal to 0.
% 
% similarity = 1 - distance


% USING MANHATTAN DISTANCE
for i=1:mTest
	% current test vector
	xi = Xtest(i,:); 
	for j=1:m
		% current teaching vector
		xj = X(j,:); 
		% calculate similarity with current test vector and teaching 
		% vector. store the similarity in resulting vector. This one has
		% to have the same number of rows as teaching set.
		similarity(j,1) = 1 - manhattan_distance(xi, xj);
	end
	% At this point of algorithm you have a vector of similarities of 
	% current test vector and all teaching vectors. Each index in 
	% similarity vector is bound to an index of coresponding teaching 
	% example. Next step is to sort the vector of similarities in 
	% descending order - from most to less similar. Do it in such 
	% fashion that memorise oryginal indexes thus indexes before sorting 
	[_, idx] = sort(similarity,'descend'); 
	% Take only firs k entries of indexes.
	idx = idx(1:k,:)';
	% Take classes of those examples, which indexes you selected during
	% limiting the sorted similarity vector to k entries
	classes = y(idx,1)
	% Since this is a binary classification you can have either ones or
	% zeros. Performing sum on extracted classes you will comput
	% the number of positive classes...
	a = sum(classes);
	% ...and subtracting it from k you get the number of negative classes
	b   = k-a;
	% Following is a voting process. Mind i which is the index of current
	% testing example.
	if (a>=b)
		% so the positive class is at least in majority, hence the i-th
		% testing example is predicted as belonging to class 1
		prediction(i)=1;
	else
		% so negative class is in majority, hence the i-th
		% testing example is predicted as belonging to class 0
		prediction(i)=0;
	end	
end
% Following
% gives 1 for true  possitive and true  negative
% gives 0 for false possitive and false negative.
tp_tn = (prediction==ytest);
% calculate how many predictions are correct
correct = sum(tp_tn);
% Hence the efficiensy in % is given by
eff = correct/mTest*100;
printf('The efficiency of kNN with Manhattan distance: %f\n',eff);

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
printf('The efficiency of kNN with Euclidean distance: %f\n',eff);

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
printf('The efficiency of kNN with Chebyshev distance: %f\n',eff);

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
printf('The efficiency of kNN with Hamming distance:   %f\n',eff);

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
printf('The efficiency of kNN with cosine similarity:  %f\n',eff);
