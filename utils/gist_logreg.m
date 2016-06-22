function [svm,LOG]=gist_logreg(X,y,options)
% GIST solver for the problem
%
%   min_x |y-Xw|^2/n+\lambda*g(x)
%
% options:
%   options.lambda : reg term (default=1)
%   options.eps : l2 reg term for bounded fun (default=1e-8)
%   options.reg: reg term (l1,lsp,l2) (default='l2')
%   options.bias: learn bias (default=1)

options=initoptions(mfilename,options);

[g,prox_g]=get_reg_prox(options.reg,options);

if options.bias
    X0=[X ones(size(X,1),1)];
    g=@(x) g(x(1:end-1,:));
    prox_g=@(x,lambda) prox_bias(x,lambda,prox_g);
else
    X0=X;
end

vals=unique(y);
nbclass=length(vals);

y0=zeros(size(X,1),nbclass);

for i=1:nbclass
    y0(y==vals(i),i)=1;
end

y0=logical(y0);

f=@(x) cost(x,X0,y0);
df=@(x) grad(x,X0,y0);

W0=zeros(size(X0,2),nbclass);


[W,LOG]=gist_opt(f,df,g,prox_g,W0,options);

if options.bias
    svm.w=W(1:end-1,:);
    svm.w0=W(end,:);
else
    svm.w=W;
    svm.w0=0;
end

svm.W=W;

svm.multiclass=1;
svm.nbclass=length(vals);
svm.vals=vals;


end

function df=grad(w,X,y)
    [R]=lr_mc_residue(w,X,y);
    df=X'*R/size(X,1);
end

function f=cost(w,X,y)
    t=size(y,2);
    M=X*w;
    py=M(y);
    E=exp(M-py*ones(1,t));
    f=sum(log(sum(E,2)))/size(X,1);
end

function [R]=lr_mc_residue(w,X,y)
t=size(y,2);
M=X*w;
py=M(y);

E=exp(M-py*ones(1,t));

SE=sum(E,2)*ones(1,t);

R=(E-y.*SE)./SE;
end

function res=prox_bias(x,lambda,prox)
    res=prox(x,lambda);
    res(end,:)=x(end,1);
end
