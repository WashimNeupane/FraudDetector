load('E:\matlab\CAB301_GROUP\DATA\credit.mat')
[m,n] = size(creditRisk);
%Count number of fraudulent transactions
n_fraud = sum(creditRisk(:,31));
n_normal = m - n_fraud;

%perntage of fraud and normal
n_fraud_rate = n_fraud/m * 100;
n_normal_rate = n_normal/m * 100;

%change time to hours instead of seconds
creditRisk(:,1) = creditRisk(:,1)/3600;

%plot amount vs class
% figure
% scatter(creditRisk(:,31),creditRisk(:,30))
% xlim([-1 2]);
% xlabel('Category'); ylabel('Amount');
% title('Fraudulent transaction per amount')

%plot histogram against time in hours
% figure
fraud = creditRisk(creditRisk(:,31)==1);
non_fraud = creditRisk(creditRisk(:,31)==0);
% histogram(fraud,'DisplayStyle','stairs');
% hold on
% histogram(non_fraud,'DisplayStyle','stairs');
% hold off
% legend('fraud','non fraud')
% title('Frequency of transactions for given time')
% xlabel('time in hours')
% ylabel('number of transactions');
% 
% %plot correlation (NOTE: uses Econometrics toolbox, takes long time to process)
% % corrplot(creditRisk);
% 
% %plot histogram of transaction amount
% figure
% histogram(creditRisk(:,30));
% title('Frequency of transaction amomunt')
% xlabel('Amounts in $')
% ylabel('number of transactions');
% xlim([0 200])
% 
% %plot rate of fraudulent to normal transaction
% figure 
% histogram(creditRisk(:,31),3);
% title('Frequency of normal to fraudulent transactions')
% xlabel('category')
% ylabel('number of transactions');

%data with all non correlated features removed:
cre = [creditRisk(:,18) creditRisk(:,15) creditRisk(:,13) creditRisk(:,11)];
cre = [cre creditRisk(:,12) creditRisk(:,5) creditRisk(:,3) creditRisk(:,20) creditRisk(:,31)];
%get random sample of data
% [training, testing] = generateTrainData(creditRisk);
[training, testing] = generateTrainData(cre);
[mm,nn] = size(training);
%k fold cross validation
k = length(training)/15;
for i=1:15
    model = fitcknn(training((k*(i-1)+1):(k*i),1:(nn-1)),training((k*(i-1) + 1):(k*i),nn),'NumNeighbors',3);
    predictval = predict(model, testing(:,1:(nn-1)));
    KNNconfusionMM = predictval - 2* testing(:,nn);
    KNNconfusionMat(i,1)=sum(sum(KNNconfusionMM(:) == 0));
    KNNconfusionMat(i,2)=sum(sum(KNNconfusionMM(:) == 1));
    KNNconfusionMat(i,3)=sum(sum(KNNconfusionMM(:) == -2));
    KNNconfusionMat(i,4)=sum(sum(KNNconfusionMM(:) == -1));
    error_knn(i) = sum(predictval ~= testing(:,nn));
end
errorrate_knn= (error_knn/164)*100;
figure
plot(errorrate_knn);
title('Error rate for Knn anomaly detector')
xlabel('Cross validation trial number')
ylabel('Error Rate');
kvalue = [2 3 5 7 10  20 30];
rr =     [9 8.47 10 11 12 16 22];
figure
plot(kvalue, rr);
title('Optimal K-value')
xlabel('K-value')
ylabel('Average Error Rate across several trials(%)');

% %k fold cross validation
% k = length(training)/15;
% for i=1:15
%     model = fitctree(training((k*(i-1)+1):(k*i),1:(nn-1)),training((k*(i-1) + 1):(k*i),nn));
%     predictval = predict(model, testing(:,1:(nn-1)));
%     DTconfusionMM = predictval - 2* testing(:,nn);
%     error_dctree(i) = sum(predictval ~= testing(:,nn));
%     confusionMat(i,1)=sum(sum(DTconfusionMM(:) == 0));
%     confusionMat(i,2)=sum(sum(DTconfusionMM(:) == 1));
%     confusionMat(i,3)=sum(sum(DTconfusionMM(:) == -2));
%     confusionMat(i,4)=sum(sum(DTconfusionMM(:) == -1));
% end
% figure
% errorrate_dctree= (error_dctree/164)*100;
% plot(errorrate_dctree);
% title('Error rate for Decision tree anomaly detector')
% xlabel('Cross validation trial number')
% ylabel('Error Rate');
% view(model,'mode','graph')
% 

for i=1:15
    model = fitcsvm(training((k*(i-1)+1):(k*i),1:(nn-1)),training((k*(i-1) + 1):(k*i),nn));
    predictval = predict(model, testing(:,1:(nn-1)));
    KNNconfusionMM = predictval - 2* testing(:,nn);
    SVMconfusionMat(i,1)=sum(sum(DTconfusionMM(:) == 0));
    SVMconfusionMat(i,2)=sum(sum(DTconfusionMM(:) == 1));
    SVMconfusionMat(i,3)=sum(sum(DTconfusionMM(:) == -2));
    SVMconfusionMat(i,4)=sum(sum(DTconfusionMM(:) == -1));
    error_SVM(i) = sum(predictval ~= testing(:,nn));
end
errorrate_SVM= (error_SVM/164)*100;
figure
plot(errorrate_SVM);
title('Error rate for SVM anomaly detector')
xlabel('Cross validation trial number')
ylabel('Error Rate');




