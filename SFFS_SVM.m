%---- SFFS ----
clear all;
close all;
clc;
%--- load data ---
load FeatureData_85_114.mat;
F_ori=FeatureData_85_114; % 85 samples, 114 kinds of features
labels=[ones(53,1);ones(32,1)+1]; % 54:disease control£¬32:treatment failure
 F=zscore(F_ori);
m=size(F,2); 

%% Start
chosen=[];% selected feature set 
gamma=0.9; % the hyperparameter gamma of SVM classifier
tic;
  %% SFS:select two features
J=zeros(1,m); 
d=0;
while d<2
     for i=1:m   
        if ~any(chosen==i)
            J(i)=svmAUC(F(:,[chosen,i]),labels,gamma);% J:classification performance(AUC)
        end    
     end             
    [ma0,we0]=max(J);
       if ~any(chosen==we0)
           chosen=[chosen we0]
           d=d+1;
       end    
end

  %% --- Select features ---   
% while d<13 %Alternatively: the limit of the lagest number of the selected features  
flag=1;% when flag==0, get out of two layers of WHILE loops
while flag
     for i=1:m   
        if ~any(chosen==i)
            J(i)=svmAUC(F(:,[chosen,i]),labels,gamma);
        end    
     end             
    [ma,we]=max(J);
       if any(chosen==we)
           break;
       else
           chosen=[chosen we];
           chosen
           d=d+1;
       end               
  %% --- remove features ---
   while size(chosen,2)>2  
    chnum=size(chosen,2); 
    J1=zeros(1,chnum);
    J1(1)=svmAUC(F(:,chosen(2:end)),labels,gamma);
    J1(chnum)=svmAUC(F(:,chosen(1:(chnum-1))),labels,gamma);
       for i=2:chnum-1   % remove the i-th feature in chosen 
          J1(i)=svmAUC(F(:,chosen([1:i-1,i+1:chnum])),labels,gamma);
       end
    [ma1,we1]=max(J1);
    %%
        if ma1>=ma
            if chosen(we1)==we
               J(we)=ma1;
               chosen(we1)=[];
               d=d-1;
               flag=0;%get out of two layers of WHILE loops
               chosen
                break; 
            else 
               J(we)=ma1;
               chosen(we1)=[];
               d=d-1;
               chosen
            end
        else 
            break;
        end
   end
end 
disp('features selection is over!');
chosen
if length(chosen)>2
    fprintf('max(J)=%2.4f  ',ma)
else
    fprintf('max(J)=%2.4f  ',ma0)
end
toc;