function AUC=svmAUC(Data,Label,gamma)
P=zeros(size(Data,1),2);
 id=1:size(Data,1);
%% k fold validation
indices = crossvalind('Kfold', size(Data,1), 10);%将数据样本随机分割为3部分
for i = 1:10 %循环10次，分别取出第i部分作为测试样本，其余部分作为训练样本
    test = (indices == i);
    train = ~test;
    trainData = Data(train, :);
    testData = Data(test, :);
    trainLabel= Label(train, :);
    testLabel= Label(test, :); 
        model = fitcsvm(trainData, trainLabel,'KernelFunction','rbf','KernelScale',gamma);
        [predict_label,prob,cost] = predict(model,testData);
   %  acc(i)=sum(testLabel==predict_label)/length(testLabel);
   P(id(test),:)=prob;
end
   % Acc=mean(acc);

%% leave one out validation
%   %  predict_label=zeros(size(Label));
%   for i=1:size(Data,1)  
%      trainid=setdiff(id,i);
%      trainData = Data(trainid,:);
%      trainLabel= Label(trainid);
%      model = fitcsvm(trainData, trainLabel,'KernelFunction','rbf','KernelScale',gamma);
%      [label,prob] = predict(model,Data(i,:));
%   %   predict_label(i)=label;
%      P(i,:)=prob;
%   end
%   %   Acc=sum(Label==predict_label)/length(Label); 
%   
 
 %%
[~,~,~,AUC] = perfcurve(Label,P(:,2),2);
end


