% test gist

clear all
close all
addpath(genpath('.'))


%% generate dataset


nbperclass=100;


% generating good features and labels
x=[];
y=[];
sigma=1;
m1=[1, .5];
m2=-m1;
x=[ones(nbperclass,1)*m1+sigma*randn(nbperclass,2);ones(nbperclass,1)*m2+sigma*randn(nbperclass,2)];
y=[ones(nbperclass,1);-ones(nbperclass,1)];


% adding random features 
nbnoise=18;
x=[x sigma*randn(size(x,1),nbnoise)];

%% visu data

figure(1)

plot(x(y==1,1),x(y==1,2),'+')
hold on
plot(x(y==-1,1),x(y==-1,2),'xr')
hold off


%% bayes decision

wb=(m1-m2)

fb=@(x,y) x*wb(1)+y*wb(2);


figure(1)

plot(x(y==1,1),x(y==1,2),'+')
hold on
plot(x(y==-1,1),x(y==-1,2),'xr')
h=ezplot(fb);
set(h, 'Color','b')
hold off
%title('test')
legend('Class 1','Class -1','Bayes decision')



%% svm l1

options.verbose=1;
options.lambda=1e-1
options.theta=.01;
options.p=.5;
options.reg='l1'

tic
[svml1,LOG]=gist_hinge2(x,y,options)
%[svm,LOG]=gist_chinge(x,y,options)
%[svml1,LOG]=gist_logreg(x,y,options)
toc

wl1=svml1.w(:,1);

fl1=@(x,y) x*wl1(1)+y*wl1(2);

figure(1)

plot(x(y==1,1),x(y==1,2),'+')
hold on
plot(x(y==-1,1),x(y==-1,2),'xr')
h=ezplot(fb);
set(h, 'Color','b')
h=ezplot(fl1);
set(h, 'Color','r')
hold off
%title('test')
legend('Class 1','Class -1','Bayes decision','l1 reg.')

%% LSP

options.verbose=1;
options.lambda=2e-3
options.theta=.001;
options.p=.5;
options.reg='lsp'

tic
[svmlsp,LOG]=gist_hinge2(x,y,options)
%[svm,LOG]=gist_chinge(x,y,options)
%[svmlsp,LOG]=gist_logreg(x,y,options)
toc

wlsp=svmlsp.w(:,1);

flsp=@(x,y) x*wlsp(1)+y*wlsp(2);
figure(1)

plot(x(y==1,1),x(y==1,2),'+')
hold on
plot(x(y==-1,1),x(y==-1,2),'xr')
h=ezplot(fb);
set(h, 'Color','b')
h=ezplot(fl1);
set(h, 'Color','r')
h=ezplot(flsp);
set(h, 'Color',[0 .7 0])
hold off
%title('test')
legend('Class 1','Class -1','Bayes decision','l1 reg.','lsp reg.')


%% lp

options.verbose=1;
options.lambda=6e-2
options.theta=.1;
options.p=.5;
options.reg='lp'

tic
[svmlp,LOG]=gist_hinge2(x,y,options)
%[svm,LOG]=gist_chinge(x,y,options)
%[svmlsp,LOG]=gist_logreg(x,y,options)
toc

limx=[-4,4]

wlp=svmlp.w(:,1);

flp=@(x,y) x*wlp(1)+y*wlp(2);
figure(1)

plot(x(y==1,1),x(y==1,2),'+')
hold on
plot(x(y==-1,1),x(y==-1,2),'xr')
h=ezplot(fb,limx);
set(h, 'Color','b')
h=ezplot(fl1,limx);
set(h, 'Color','r')
h=ezplot(flsp,limx);
set(h, 'Color',[0 .7 0],'LineStyle','-.')
h=ezplot(flp,limx);
set(h, 'Color',[0 .7 .7],'LineStyle','--')
hold off
title('')
legend('Class 1','Class -1','Bayes decision','l1 reg.','lsp reg.','lp reg.')

print('-depsc','toybias.eps')