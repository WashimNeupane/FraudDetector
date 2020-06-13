function [training, testing] = generateTrainData(creditRisk)
[m,n] = size(creditRisk);

fraudSample = creditRisk(:,n)==1;
normalSample = creditRisk(:,n)==0;
fraudSample  = creditRisk((fraudSample),:);
normalSample = creditRisk((normalSample),:);

partitionNormalSample = cvpartition(normalSample(1:902,n),'holdout',820);
partitionfraudSample =  cvpartition(fraudSample(:,n),'holdout',410);

testCat1 = fraudSample(partitionfraudSample.training,:);
trainCat1 = fraudSample(partitionfraudSample.test,:);

testCat2 = normalSample(partitionNormalSample.training,:);
trainCat2 = normalSample(partitionNormalSample.test,:);

training = [trainCat1; trainCat2];
testing  = [testCat1 ;testCat2];

training = training(randperm(size(training,1)),:);
testing = testing(randperm(size(testing,1)),:);

end