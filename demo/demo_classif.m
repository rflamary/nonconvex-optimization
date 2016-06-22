% This file show how to use the toolbox to estimate linear classifier with
% different regularization scheme.
% note that all classifiers naturally handle multiclass data
%
% The dataset contains only 2 discriminant dimensions and 8 noisy
% dimension.
% The timated classifier should select automatically the two first
% dimension when sparsity is promoted.

clear all
close all
addpath(genpath('.'))


%% generate dataset


mclass=[1 1; -1 1; 1 -1; -1 -1];

nbperclass=1000;


% generating good features and labels
x=[];
y=[];
sigma=.5;
for i=1:size(mclass,1);
   
    x=[x; ones(nbperclass,1)*mclass(i,:)+sigma*randn(nbperclass,size(mclass,2))];
    y=[y;i*ones(nbperclass,1)];
end


% adding random features 
nbnoise=8;
x=[x sigma*randn(size(x,1),nbnoise)];

% models should have only the two first components active

%% visu data on 2 diuscriminant components

figure(1)

plot(x(y==1,1),x(y==1,2),'+')
hold on
plot(x(y==2,1),x(y==2,2),'x')
plot(x(y==3,1),x(y==3,2),'o')
plot(x(y==4,1),x(y==4,2),'s')
hold off

%% SVM with l2 regularization

% options for solver
options.verbose=1;
options.lambda=1e-3 ;% regul parameter
options.theta=.01; % parameter for lsp
options.p=.5; % parameter for lp
options.reg='l2'; % l2 l1 lp, lsp are possible options

tic
[svml2]=gist_hinge2(x,y,options); % hinge squared svn
%[svm,LOG]=gist_chinge(x,y,options) % calibrated hinge
%[svm,LOG]=gist_logreg(x,y,options) % logistic regression
toc

% classifier linear parameter
wl2=svml2.w % normal vector to hyperlpaln
w0l2=svml2.w0; % svm bias

%% SVM with lp regularization

% options for solver
options.verbose=1;
options.lambda=1e-2 ;% regul parameter
options.theta=.01; % parameter for lsp
options.p=.5; % parameter for lp
options.reg='lp'; % l2 l1 lp, lsp are possible options

tic
[svmlp]=gist_hinge2(x,y,options); % hinge squared svn
%[svm,LOG]=gist_chinge(x,y,options) % calibrated hinge
%[svm,LOG]=gist_logreg(x,y,options) % logistic regression
toc

% classifier linear parameter
wlp=svmlp.w % normal vector to hyperlpaln
w0lp=svmlp.w0; % svm bias



%% visu separation
% plot classification regions in 2D (discriminant features)

nbgrid=100;
[Xgrid,Ygrid]=meshgrid(linspace(-3,3,nbgrid),linspace(-3,3,nbgrid));

xtest=[Xgrid(:) Ygrid(:)];

ypred=xtest*wlp(1:2,:)+ones(nbgrid*nbgrid,1)*w0lp;

[temp,ypred_c]=min(ypred,[],2);

Ypred=reshape(ypred_c,[nbgrid nbgrid]);

figure(2)

imagesc(linspace(-3,3,nbgrid),linspace(-3,3,nbgrid),Ypred)

hold on
plot(x(y==1,1),x(y==1,2),'+')
plot(x(y==2,1),x(y==2,2),'x')
plot(x(y==3,1),x(y==3,2),'o')
plot(x(y==4,1),x(y==4,2),'s')
hold off


