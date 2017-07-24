clear all;
close all,
clc;

data = load('../../datasets/hash-recogintion/dataset.csv');
data = [ones(4000,1) data];

X=data(1:2800, 1:9);
y=data(1:2800, 10);

Xtest=data(2801:end, 1:9);
ytest=data(2801:end, 10);

k = 3;
[m_test,n_test]=size(Xtest);

%p = zeros(m_test,1);
%for i=1:m_test
%	x = Xtest(i,:);
%	similarity = euclidean_distance(X, x);
%	[a, index] = sort(similarity,'ascend');
%	index = index(1:k,:)';
%	a = sum(y(index,1));
%	b = k-a;
%	if (a>=b)
%		p(i)=1;
%	else
%		p(i)=0;
%	end
%end
%sum((p==ytest))/m_test*100

p = zeros(m_test,1);
for i=1:m_test
	x = Xtest(i,:);
	similarity = cosine_similarity(X, x);
	[_, index] = sort(similarity,'descend');
	index = index(1:k,:)';
	a = sum(y(index,1));
	b = k-a;
	if (a>=b)
		p(i)=1;
	else
		p(i)=0;
	end	
end
sum((p==ytest))/m_test*100
