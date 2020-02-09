% Expected risk minimization with 2 classes
clear all, close all,

n = 2; % number of feature dimensions
N = 10000; % number of iid samples
mu(:,1) = [-0.1;0]; mu(:,2) = [0.1;0];
Sigma(:,:,1) = [1 -0.9;-0.9 1]; Sigma(:,:,2) = [1 0.9;0.9 1];
p = [0.8 0.2]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1); %generate N number of sample and compare with 0.8, if >= 0.8, then it falls in label 1 region, label 0 region otherwise.
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class, Nc(1) is # of label 0, Nc(2) is # of label 1.
x = zeros(n,N); % save up space, 2x10000 of zeros (nxN)
lambda = [0 1;1 0];
% Draw samples from each class pdf
for L = 0:1
    x(:,label==L) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc(L+1))'; %x(:,label==L):store generated column vector in where label is 0 or 1, very compact. Each column in x is a mvrnd vector.
end
%graph generated class 0 and class 1 vector
figure(1), clf, %create fig 2 and clear fig window
plot(x(1,label==0),x(2,label==0),'o'), hold on, %hold on retains plots in the current axes so that new plots added to the axes do not delete existing plots. 
plot(x(1,label==1),x(2,label==1),'+'), axis equal, %axis equal sets the aspect ratio so that the data units are the same in every direction.
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'),

discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));
sortedScores = sort(discriminantScore);
tau = zeros(1+length(sortedScores));
epsilon = 0.01;
tau(1) = sortedScores(1) - epsilon; % min of tau
tau(length(tau)) = sortedScores(length(sortedScores)) + epsilon; % max of tau
for i = 2:length(tau)-1
    tau(i) = (sortedScores(i) + sortedScores(i-1))/2;
end

ROC = zeros(4,length(tau));
for i = 1:length(tau)
    decision = (discriminantScore >= tau(i));
    ind00 = find(decision==0 & label==0); % index of 00, correct decision on 0 label
    p00 = length(ind00)/Nc(1); % probability of true negative, P(x=0|L=0)
    ind10 = find(decision==1 & label==0);
    p10 = length(ind10)/Nc(1); % probability of false positive
    ind01 = find(decision==0 & label==1);
    p01 = length(ind01)/Nc(2); % probability of false negative
    ind11 = find(decision==1 & label==1);
    p11 = length(ind11)/Nc(2); % probability of true positive
    pError = [p10,p01]*Nc'/N; % probability of error, empirically estimated
    ROC(:,i) = [p11; p10; pError; tau(i)];
end

[~,indexMinPerror] = min(ROC(3,:));
disp('Theoretical optimal tau - ERM: '); disp(log((lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2)));
disp('Empirical optimal tau - ERM: '); disp(ROC(4,indexMinPerror(1)));
disp('Empirical min probability of error - ERM: '); disp(ROC(3,indexMinPerror(1)));

figure(2),
plot(ROC(2,:),ROC(1,:)); hold on,
plot(ROC(2,indexMinPerror(1)),ROC(1,indexMinPerror(1)),'og'); hold on,
title('Detection vs. false alarm ROC - ERM'),
xlabel('False alarm'), ylabel('Detection'),
legend('Detection vs. false alarm','Minimun probability of error point'),

figure(3),
plot3(ROC(2,:),ROC(1,:), ROC(4,:)); hold on,
plot3(ROC(2,indexMinPerror(1)),ROC(1,indexMinPerror(1)),ROC(4,indexMinPerror(1)),'og'); hold on,
title('Tau vs. detection vs. false alarm ROC - ERM'),
xlabel('False alarm'), ylabel('Detection'), zlabel('Tau')
legend('Tau vs. detection vs. false alarm','Minimun probability of error point'),

% LDA
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend'); % get index of eigen vector with max variance
wLDA = V(:,ind(1)); % Fisher LDA projection vector = eigen vector with max variance
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

sortedScores = sort(yLDA);
tau = zeros(1+length(sortedScores));
epsilon = 0.01;
tau(1) = sortedScores(1) - epsilon;
tau(length(tau)) = sortedScores(length(sortedScores)) + epsilon;
for i = 2:length(tau)-1
    tau(i) = (sortedScores(i) + sortedScores(i-1))/2;
end

ROC = zeros(4,length(tau));
for i = 1:length(tau)
    decisionLDA = (yLDA >= tau(i));
    ind00 = find(decisionLDA==0 & label==0); % index of 00, correct decision on 0 label
    p00 = length(ind00)/Nc(1); % probability of true negative, P(x=0|L=0)
    ind10 = find(decisionLDA==1 & label==0);
    p10 = length(ind10)/Nc(1); % probability of false positive
    ind01 = find(decisionLDA==0 & label==1);
    p01 = length(ind01)/Nc(2); % probability of false negative
    ind11 = find(decisionLDA==1 & label==1);
    p11 = length(ind11)/Nc(2); % probability of true positive
    pError = [p10,p01]*Nc'/N; % probability of error, empirically estimated
    ROC(:,i) = [p11; p10; pError; tau(i)];
end

[~,indexMinPerror] = min(ROC(3,:));
disp('Empirical optimal tau - LDA: '); disp(ROC(4,indexMinPerror(1)));
disp('Empirical min probability of error - LDA: '); disp(ROC(3,indexMinPerror(1)));

figure(4), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal, hold on,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

figure(5),
plot(ROC(2,:),ROC(1,:)); hold on,
plot(ROC(2,indexMinPerror(1)),ROC(1,indexMinPerror(1)),'og'); hold on,
title('Detection vs. false alarm - LDA'),
xlabel('False alarm'), ylabel('Detection'),
legend('Detection vs. false alarm','Minimun probability of error point'),

figure(6),
plot3(ROC(2,:),ROC(1,:), ROC(4,:)); hold on,
plot3(ROC(2,indexMinPerror(1)),ROC(1,indexMinPerror(1)),ROC(4,indexMinPerror(1)),'og'); hold on,
title('Tau vs. detection vs. false alarm - LDA'),
xlabel('False alarm'), ylabel('Detection'), zlabel('Tau')
legend('Tau vs. detection vs. false alarm','Minimun probability of error point'),


%NaiveBayesian
discriminantScore = log(evalGaussian(x,mu(:,2),eye(2)))-log(evalGaussian(x,mu(:,1),eye(2)));
sortedScores = sort(discriminantScore);
tau = zeros(1+length(sortedScores));
epsilon = 0.01;
tau(1) = sortedScores(1) - epsilon;
tau(length(tau)) = sortedScores(length(sortedScores)) + epsilon;
for i = 2:length(tau)-1
    tau(i) = (sortedScores(i) + sortedScores(i-1))/2;
end

ROC = zeros(4,length(tau));
for i = 1:length(tau)
    decision = (discriminantScore >= tau(i));
    ind00 = find(decision==0 & label==0); % index of 00, correct decision on 0 label
    p00 = length(ind00)/Nc(1); % probability of true negative, P(x=0|L=0)
    ind10 = find(decision==1 & label==0);
    p10 = length(ind10)/Nc(1); % probability of false positive
    ind01 = find(decision==0 & label==1);
    p01 = length(ind01)/Nc(2); % probability of false negative
    ind11 = find(decision==1 & label==1);
    p11 = length(ind11)/Nc(2); % probability of true positive
    pError = [p10,p01]*Nc'/N; % probability of error, empirically estimated
    ROC(:,i) = [p11; p10; pError; tau(i)];
end

[~,indexMinPerror] = min(ROC(3,:));
disp('Empirical optimal tau - NB: '); disp(ROC(4,indexMinPerror(1)));
disp('Empirical min probability of error - NB: '); disp(ROC(3,indexMinPerror(1)));

figure(7),
plot(ROC(2,:),ROC(1,:)); hold on,
plot(ROC(2,indexMinPerror(1)),ROC(1,indexMinPerror(1)),'og'); hold on,
title('Detection vs. false alarm - Naive Bayesian'),
xlabel('False alarm'), ylabel('Detection'),
legend('Detection vs. false alarm','Minimun probability of error point'),

figure(8),
plot3(ROC(2,:),ROC(1,:), ROC(4,:)); hold on,
plot3(ROC(2,indexMinPerror(1)),ROC(1,indexMinPerror(1)),ROC(4,indexMinPerror(1)),'og'); hold on,
title('Tau vs. detection vs. false alarm - Naive Bayesian'),
xlabel('False alarm'), ylabel('Detection'), zlabel('Tau')
legend('Tau vs. detection vs. false alarm','Minimun probability of error point'),

%LDA
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = eye(2) + eye(2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

sortedScores = sort(yLDA);
tau = zeros(1+length(sortedScores));
epsilon = 0.01;
tau(1) = sortedScores(1) - epsilon;
tau(length(tau)) = sortedScores(length(sortedScores)) + epsilon;
for i = 2:length(tau)-1
    tau(i) = (sortedScores(i) + sortedScores(i-1))/2;
end

ROC = zeros(4,length(tau));
for i = 1:length(tau)
    decisionLDA = (yLDA >= tau(i));
    ind00 = find(decisionLDA==0 & label==0); % index of 00, correct decision on 0 label
    p00 = length(ind00)/Nc(1); % probability of true negative, P(x=0|L=0)
    ind10 = find(decisionLDA==1 & label==0);
    p10 = length(ind10)/Nc(1); % probability of false positive
    ind01 = find(decisionLDA==0 & label==1);
    p01 = length(ind01)/Nc(2); % probability of false negative
    ind11 = find(decisionLDA==1 & label==1);
    p11 = length(ind11)/Nc(2); % probability of true positive
    pError = [p10,p01]*Nc'/N; % probability of error, empirically estimated
    ROC(:,i) = [p11; p10; pError; tau(i)];
end

[~,indexMinPerror] = min(ROC(3,:));
disp('Empirical optimal tau - NB/LDA: '); disp(ROC(4,indexMinPerror(1)));
disp('Empirical min probability of error - NB/LDA: '); disp(ROC(3,indexMinPerror(1)));

figure(9), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal, hold on,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels - Naive Bayesian'),
xlabel('x_1'), ylabel('x_2'), 

figure(10),
plot(ROC(2,:),ROC(1,:)); hold on,
plot(ROC(2,indexMinPerror(1)),ROC(1,indexMinPerror(1)),'og'); hold on,
title('Detection vs. false alarm - LDA + Naive Bayesian'),
xlabel('False alarm'), ylabel('Detection'),
legend('Detection vs. false alarm','Minimun probability of error point'),

figure(11),
plot3(ROC(2,:),ROC(1,:), ROC(4,:)); hold on,
plot3(ROC(2,indexMinPerror(1)),ROC(1,indexMinPerror(1)),ROC(4,indexMinPerror(1)),'og'); hold on,
title('Tau vs. detection vs. false alarm - LDA + Naive Bayesian'),
xlabel('False alarm'), ylabel('Detection'), zlabel('Tau')
legend('Tau vs. detection vs. false alarm','Minimun probability of error point'),