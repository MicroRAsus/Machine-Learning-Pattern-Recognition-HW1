% Expected risk minimization with 2 classes and mixed Gaussians
clear all, close all,

n = 2;      % number of feature dimensions
N = 1000;   % number of iid samples

% Class 0 parameters (2 gaussians)
mu(:,1) = [-2;1]; mu(:,2) = [0;1];
Sigma(:,:,1) = [3 2;2 3]/8; Sigma(:,:,2) = [5 -1;-1 10]/12;
p0 = [0.7 0.3]; % probability split between 2 distributions

% Class 1 parameters (2 gaussians)
mu(:,3) = [5;1]; mu(:,4) = [3;-2]; 
Sigma(:,:,3) = [2 -2;-2 5]/10; Sigma(:,:,4) = [20 1;1 5]/14;
p1 = [0.15 0.85]; % probability split between 2 distributions

p = [0.35,0.65]; % class priors for labels 0 and 1 respectively
label = rand(1,N) > p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space

% Draw samples from each class pdf
for i = 1:N
    if label(i) == 0
        % Class 0 samples for each gaussian based on their distribution
        distr = rand > p0(1);
        if distr
            x(:,i) = mvnrnd(mu(:,2),Sigma(:,:,2),1)';
        else
            x(:,i) = mvnrnd(mu(:,1),Sigma(:,:,1),1)';
        end
   
    else
        % Class 1 samples for each gaussian based on their distribution
        distr = rand > p1(1);
        if distr
            x(:,i) = mvnrnd(mu(:,4),Sigma(:,:,4),1)';
        else
            x(:,i) = mvnrnd(mu(:,3),Sigma(:,:,3),1)';
        end
    end
end

% Plot with class labels
figure(1), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2')

% calculate threshold based on loss values
lambda = [0 1;1 0];
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); 

% calculate discriminant score based on class pdfs
class0pdf = p0(1)*evalGaussian(x,mu(:,1),Sigma(:,:,1)) + p0(2)*evalGaussian(x,mu(:,2),Sigma(:,:,2)); %p(00)g(x|mu00,sigma00)+p(01)g(x|mu01,sigma01)=P_0(x)
class1pdf = p1(1)*evalGaussian(x,mu(:,3),Sigma(:,:,3)) + p1(2)*evalGaussian(x,mu(:,4),Sigma(:,:,4)); %p(10)g(x|mu10,sigma10)+p(11)g(x|mu11,sigma11)=P_1(x)
discriminantScore = log(class1pdf)-log(class0pdf);

% compare score to threshold to make decisions
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive

pError = [p10,p01]*Nc'/N;
disp('Empirical probability of error: '); disp(pError);

% plot correct and incorrect decisions
figure(2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,3),Sigma(:,:,3))+ evalGaussian([h(:)';v(:)'],mu(:,4),Sigma(:,:,4)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))+evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(2), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]), % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function'), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 

% sortedScores = sort(discriminantScore);
% tau = zeros(1+length(sortedScores));
% epsilon = 0.01;
% tau(1) = sortedScores(1) - epsilon; % min of tau
% tau(length(tau)) = sortedScores(length(sortedScores)) + epsilon; % max of tau
% for i = 2:length(tau)-1
%     tau(i) = (sortedScores(i) + sortedScores(i-1))/2;
% end
% 
% ROC = zeros(4,length(tau));
% for i = 1:length(tau)
%     decision = (discriminantScore >= tau(i));
%     ind00 = find(decision==0 & label==0); % index of 00, correct decision on 0 label
%     p00 = length(ind00)/Nc(1); % probability of true negative, P(x=0|L=0)
%     ind10 = find(decision==1 & label==0);
%     p10 = length(ind10)/Nc(1); % probability of false positive
%     ind01 = find(decision==0 & label==1);
%     p01 = length(ind01)/Nc(2); % probability of false negative
%     ind11 = find(decision==1 & label==1);
%     p11 = length(ind11)/Nc(2); % probability of true positive
%     pError = [p10,p01]*Nc'/N; % probability of error, empirically estimated
%     ROC(:,i) = [p11; p10; pError; tau(i)];
% end
% 
% [~,indexMinPerror] = min(ROC(3,:));
% disp('Optimal tau:');
% disp(ROC(3,indexMinPerror(1)));