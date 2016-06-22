% function [w,w0,LOG]=gist_least(X,y,options)
function [svm,LOG]=gist_least(X,y,options)
% GIST solver for the problem
%
%   min_x |y-Xw|^2/n+\lambda*g(x)
%
% options:
%   options.lambda : reg term (default=1)
%   options.reg: reg term (l1,lsp,l2) (default='l2')
%   options.bias : estimate a bias (default=1)
%   options.pos : force positive 0 (default=0)
%   options.W0 : force positive 0 (default=[])

options=initoptions(mfilename,options);


% get proximal operator and reg function
[g,prox_g]=get_reg_prox(options.reg,options);

% generate matrix with without bias
if options.bias
    X0=[X ones(size(X,1),1)];
    g=@(x) g(x(1:end-1,:));
    prox_g=@(x,lambda) prox_bias(x,lambda,prox_g);
else
    X0=X;
end

% add positivity constraints
if options.pos 
    prox_g=@(x,lambda) prox_g(max(x,0),lambda);
end

if isempty(options.W0)
    W0=zeros(size(X0,2),size(y,2)); % Starting point a parametrer ?
    W0(end)=0;
else
    W0=options.W0; 
end

f=@(x) cost(x,X0,y);
df=@(x) grad(x,X0,y);

[W,LOG]=gist_opt(f,df,g,prox_g,W0,options);

if options.bias
    svm.w=W(1:end-1,:);
    svm.w0=W(end,:);
else
    svm.w=W;
    svm.w0=0;
end

svm.W=W;


end

function df=grad(w,X,y)
    df=-X'*(y-X*w);%/size(X,1);
end

function f=cost(w,X,y)
    f=norm(y-X*w,'fro')^2/2;%/size(X,1)/2;
end

function res=prox_bias(x,lambda,prox)
    res=prox(x,lambda);
    res(end,:)=x(end,1);
end