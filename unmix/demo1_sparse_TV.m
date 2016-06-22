%% demo_sunsal_TV
%
% This demo illustrates the sunsal_TV sparse regression algorithm
% introduced in the paper 
%
%  M.-D. Iordache, J. Bioucas-Dias, and A. Plaza, "Total variation spatial 
%  regularization for sparse hyperspectral unmixing", IEEE Transactions on 
%  Geoscience and Remote Sensing, vol. PP, no. 99, pp. 1-19, 2012.
%
% which solves the optimization problem
%
%   min  0.5*||AX-Y||^2_F + lambda_1 ||X||_{1,1} + lambda_tv TV(X) 
%   X>=0
%
%
%  Demo parameters:
%     p = 5                             % number of endmembers
%     SNR = 40 dB      
%     size(A) = [220, 240]              % size of the library
%     min angle(a_i, a_j) = 4.44 degs   % minimum angle between any two
%                                       % elements of A
%       
%  Notes:
%
%    You may change the demo parameters, namely SNR, the noise correlation,
%    the size of dictionary A by changing min_angle, and the true endmember 
%    matrix M, which, in any case, must contain p=5 columns. 
% 
%   Please keep in mind the following:
%
%     a) sunsal  adapts automatically  the ADMM parameter for 
%        convergence speed 
%  
%     b) sunsal_tv deoes not adapts automatically  the ADMM parameter. 
%        So the inputted parameter mu has a  critical impact on the
%        convergence speed
%
%     c) the regularization parameters  were hand tuned for optimal
%        performance.
% 
% Author: Jose Bioucas Dias, August 2012

close all
clear all

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of end members
p = 5;  % fixed for this demo

%SNR in dB
SNR = 40;
% noise bandwidth in pixels of the noise  low pass filter (Gaussian)
bandwidth = 1000; % 10000 == iid noise
%bandwidth = 5*pi/224; % colored noise 


% define random states
rand('state',10);
randn('state',10);


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gererate fractional abundances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% pure pixels
x1 = eye(p);

% mixtures with two materials
x2 = x1 + circshift(eye(p),[1 0]);

% mixtures with three materials
x3 = x2 + circshift(eye(p),[2 0]);

% mixtures with four  materials
x4 = x3 + circshift(eye(p),[3 0]);

% mixtures with four  materials
x5 = x4 + circshift(eye(p),[4 0]);


% normalize
x2 = x2/2;
x3 = x3/3;
x4 = x4/4;
x5 = x5/5;


% background (random mixture)
%x6 = dirichlet(ones(p,1),1)';
x6 = [0.1149 0.0741  0.2003 0.2055, 0.4051]';   % as in the paper

% build a matrix
xt = [x1 x2 x3 x4 x5 x6];


% build image of indices to xt
imp = zeros(3);
imp(2,2)=1;

imind = [imp*1  imp*2 imp* 3 imp*4 imp*5;
    imp*6  imp*7 imp* 8 imp*9 imp*10;
    imp*11  imp*12 imp*13 imp*14 imp*15;
    imp*16  imp*17 imp* 18 imp*19 imp*20;
    imp*21  imp*22 imp* 23 imp*24 imp*25];

imind = kron(imind,ones(5));

% set backround index
imind(imind == 0) = 26;

% generare frectional abundances for all pixels
[nl,nc] = size(imind);
np = nl*nc;     % number of pixels
for i=1:np
    X(:,i) = xt(:,imind(i));
end

Xim = reshape(X',nl,nc,p);

%  image endmember 1
figure(1)
imagesc(Xim(:,:,5))
title('Frational abundance of endmember 5')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% buid the dictionary 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load USGS_1995_Library.mat
%  order bands by increasing wavelength
[dummy index] = sort(datalib(:,1));
A =  datalib(index,4:end);

% prune the library 
% min angle (in degres) between any two signatures 
% the larger min_angle the easier is the sparse regression problem
min_angle = 4.44;       
A = prune_library(A,min_angle); % 240  signature 

% order  the columns of A by decreasing angles 
[A, index, angles] = sort_library_by_angle(A);


%% select p endmembers  from A
%

% angles (a_1,a_j) \simeq min_angle)
supp = 1:p;
M = A(:,supp);
[L,p] = size(M);  % L = number of bands; p = number of material


%%
%---------------------------------
% generate  the observed  data X
%---------------------------------

% set noise standard deviation
sigma = sqrt(sum(sum((M*X).^2))/np/L/10^(SNR/10));
% generate Gaussian iid noise
noise = sigma*randn(L,np);


% make noise correlated by low pass filtering
% low pass filter (Gaussian)
filter_coef = exp(-(0:L-1).^2/2/bandwidth.^2)';
scale = sqrt(L/sum(filter_coef.^2));
filter_coef = scale*filter_coef;
noise = idct(dct(noise).*repmat(filter_coef,1,np));

%  observed spectral vector
Y = M*X + noise;


% create  true X wrt  the library A
n = size(A,2);
N = nl*nc;
XT = zeros(n,N);
XT(supp,:) = X;


%% estimate noise and filter it out
% [w,Rw] = estNoise(Y);
% 
% % determine signal subspace
% [kp,Ek] = hysime(Y,w,Rw);
% 
% % remove noise
% Y = Y-w;
% 
% % project observed data on the signal subspace
% Y = Ek*Ek'*Y;
% 
% clear w;


%%
%--------------------------------------------------------------------------
% SUNSAL and SUNSAL_TV solutions
%--------------------------------------------------------------------------

% constrained least squares CLS
lambda = 0;
[X_hat_cls] =  sunsal(A,Y,'lambda',lambda,'ADDONE','no','POSITIVITY','yes', ...
                    'TOL',1e-4, 'AL_iters',2000,'verbose','yes');

SRE_cls = 20*log10(norm(XT,'fro')/norm(X_hat_cls-XT,'fro')); 
                
                
% constrained least squares l2-l1                
lambda = 1e-2;
[X_hat_l1] =  sunsal(A,Y,'lambda',lambda,'ADDONE','no','POSITIVITY','yes', ...
                    'TOL',1e-4, 'AL_iters',2000,'verbose','yes');

SRE_l1 = 20*log10(norm(XT,'fro')/norm(X_hat_l1-XT,'fro')); 
              
                
% constrained least squares l2-l1-TV (nonisotropic)              
lambda = 1e-3;
lambda_TV = 3e-3;
[X_hat_tv_ni,res,rmse_ni] = sunsal_tv(A,Y,'MU',0.05,'POSITIVITY','yes','ADDONE','no', ...
                               'LAMBDA_1',lambda,'LAMBDA_TV', lambda_TV, 'TV_TYPE','niso',...
                               'IM_SIZE',[75,75],'AL_ITERS',200, 'TRUE_X', XT,  'VERBOSE','yes');
                          
SRE_tv_ni = 20*log10(norm(XT,'fro')/norm(X_hat_tv_ni-XT,'fro')); 

% constrained least squares l2-l1-TV (isotropic)              
lambda = 1e-3;
lambda_TV = 3e-3;
[X_hat_tv_i,res,rmse_i] = sunsal_tv(A,Y,'MU',0.05,'POSITIVITY','yes','ADDONE','no', ...
                               'LAMBDA_1',lambda,'LAMBDA_TV', lambda_TV, 'TV_TYPE','iso',...
                               'IM_SIZE',[75,75],'AL_ITERS',200, 'TRUE_X', XT,  'VERBOSE','yes');
                          
SRE_tv_i = 20*log10(norm(XT,'fro')/norm(X_hat_tv_i-XT,'fro')); 
                   

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% print results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n\n SIGNAL-TO-RECONSTRUCTION ERRORS (SRE)\n\n')

fprintf('SRE-cls = %2.3f\nSRE-l1 = %2.3f\nSRE_tv-ni = %2.3f\nSRE-tv-i = %2.3f\n\n', ...
            SRE_cls, SRE_l1,SRE_tv_ni, SRE_tv_i)
       
        
% endmember no. 1 (cls)
X_hat_cls_im = reshape(X_hat_cls', nl,nc,n);       
figure(2)
imagesc(X_hat_cls_im(:,:,supp(5)))
title('CLS - Frational abundance of endmember 5')


% endmember no. 1 (l2-l1)
X_hat_l1_im = reshape(X_hat_l1', nl,nc,n);       
figure(3)
imagesc(X_hat_l1_im(:,:,supp(5)))
title('SUnSAL - Frational abundance of endmember 5')


% endmember no. 1 (tv_ni)
X_hat_tv_ni_im = reshape(X_hat_tv_ni', nl,nc,n);       
figure(4)
imagesc(X_hat_tv_ni_im(:,:,supp(5)))
title('SUnSAL-TV (NISO) - Frational abundance of endmember 5')


% endmember no. 1 (tv_ni)
X_hat_tv_i_im = reshape(X_hat_tv_i', nl,nc,n);       
figure(5)
imagesc(X_hat_tv_i_im(:,:,supp(5)))
title('SUnSAL-TV (ISO) - Frational abundance of endmember 5')



scrsz = get(0,'ScreenSize');
figure('Position',[1 1 scrsz(3)/2 scrsz(4)/2])

subplot(151)
imagesc(XT(:,1:100))
title('spectral vectors (1;100)')

subplot(152)
imagesc(X_hat_cls(:,1:100))
axis off
title('CLS')

subplot(153)
imagesc(X_hat_l1(:,1:100))
axis off
title('SUnSAL')

subplot(154)
imagesc(X_hat_tv_ni(:,1:100))
axis off
title('SUnSAL-TV-NISO')

subplot(155)
imagesc(X_hat_tv_i(:,1:100))
axis off
title('SUnSAL-TV-ISO')



