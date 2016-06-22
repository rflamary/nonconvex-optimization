% This file show how to use the toolbox to estimate linear mixture with
% different regularization scheme.
%
% The dataset contains only 3 active components
% We compare l2 unmixing with positivity constraints, l1 and lp
% regularization

clear all
close all
addpath(genpath('.'))


%% generate data
seed=0
rng(seed)

load USGS_1995_Library.mat

D=datalib(:,4:end);

anglemin=20;
D = prune_library(D,anglemin);


wl=datalib(:,1);


d=size(D,2);
n=size(D,1);


alpha_t=zeros(d,1);


nbactive=3;

perm=randperm(d);
alpha_t(perm(1:nbactive))=rand(nbactive,1);
%wtrue=wtrue./sum(wtrue)

sigma=1e-1;

y=D*alpha_t+sigma*randn(n,1);
ytrue=D*alpha_t;

%%
figure(1)
subplot(2,1,1)

plot(wl,D)

xlabel('Wavelength in microns')

subplot(2,1,2)

plot(wl,[ytrue y])

xlabel('Wavelength in microns')

%% l2 unmix
lambda=1e-2;

alpha_l2=l2_unmix(y,D,lambda)
err_l2=sum(abs(alpha_l2-alpha_t).^2)


%% l1 unmixing

options.verbose=0; % do not print
options.lambda=1e-3;% regul parameter
options.reg='l1' % regularization
options.bias=0; % forc no bias estimation
options.pos=1; % fore positivity
options.stopvarx=1e-6; % convergence conditions
options.stopvarj=1e-6;% convergence conditions
options.nbitermax=1e4;% convergence conditions

tic
[svm_l1,LOG]=gist_least(D,y,options);
toc

alpha_l1=svm_l1.w



err_l2
err_l1=sum(abs(alpha_l1-alpha_t).^2)/2

%% lp unmixing

options.verbose=0;
options.lambda=5e-2% regul parameter
options.p=.5; % value for p
options.reg='lp'
options.bias=0; % force no bias estimation
options.pos=1; % force positivity
options.stopvarx=1e-6; % convergence conditions
options.stopvarj=1e-6;% convergence conditions
options.nbitermax=1e4;% convergence conditions

tic
[svm_lp,LOG]=gist_least(D,y,options);
toc

alpha_lp=svm_lp.w


% display errors
err_l2
err_l1
err_lp=sum(abs(alpha_lp-alpha_t).^2)/2


%% show reconsruction


figure(2)
imagesc([alpha_t alpha_l2 alpha_l1 alpha_lp]')
set(gca,'Ytick',[1 2 3 4])
set(gca,'YtickLabel',{'Ground truth','l2 unmix','l1 unmix','lp unmix'})
colorbar()



