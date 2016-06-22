function [x,LOG]=gist_opt(f,df,g,prox_g,x0,options)
% GIST solver for the problem
%
%   min_x f(x)+\lambda g(x)
%
% f     : cost function
% df    : gradien of the cost function
% g     : reg function
% prox_g: proximal function
% x0    : starting point
%
% options:
%   options.lambda : reg term (default=1e0)
%   options.eta: backward param for linesearch (default=2)
%   options.t0 : initial step (default=1)
%   options.sigma : line search param (default=1e-5)
%   options.m : line serarch param 2 (default=5)
%   options.nbitermax: max number iterations (default=1000)
%   options.stopvarx: stop threshold variation w (default=1e-5)
%   options.stopvarj: stop threshold variation cost (default=1e-5)
%   options.nbinneritermax: max number iterations (default=20)
%   options.verbose: print infos (default=0)


options=initoptions(mfilename,options);

x=x0;

grad=df(x);

loss=f(x)+options.lambda*g(x);

t=options.t0;


if options.verbose
   fprintf('|%5s|%13s|%13s|%13s|\n-------------------------------------------------\n','Iter','Loss','Dloss','Step') 
   fprintf('|%5d|%+8e|%+8e|%+8e|\n',0,loss(end),0,1/t) 
end

loop=1;
it=1;
test = 0 ;

while loop
    
    x_1=x;
    grad_1=grad;
    
    grad=df(x);
    
    x=prox_g(x_1-grad/t,options.lambda/t);
    
    loss=[loss;f(x)+options.lambda*g(x)];
    
    it2=1;
    ifin = length(loss) ;
    thr_back=max(loss(max(ifin-options.m,1):ifin-1)-options.sigma/2*t*norm(x-x_1,'fro')^2);
    while loss(end)>thr_back && it2 < options.nbinneritermax
        t=t*options.eta;
        x=prox_g(x_1-grad/t,options.lambda/t);
        loss(end)=f(x)+options.lambda*g(x);
        ifin = length(loss) ;
        thr_back=max(loss(max(ifin-options.m,1):ifin-1)-options.sigma/2*t*norm(x-x_1,'fro')^2);
        it2=it2+1;
    end
    
    xbb=x-x_1;
    ybb=grad-grad_1;
%    if it>=1 && norm(xbb,'fro')>1e-12 && norm(ybb,'fro')>1e-12
    if it>=1 && norm(xbb,'fro')/size(xbb,1)>1e-12 && norm(ybb,'fro')/size(ybb,1)>1e-12
        t=abs(sum(sum((xbb.*ybb)))/sum(sum(xbb.*xbb)));
        t = min(max(t,1e-20),1e20);
    end
    
    if options.verbose
       if mod(it,20)==0
           fprintf('|%5s|%13s|%13s|%13s|\n-------------------------------------------------\n','Iter','Loss','Dloss','Step') 
       end
       fprintf('|%5d|%+8e|%+8e|%+8e|\n',it,loss(end),(loss(end)-loss(end-1))/abs(loss(end-1)),1/t) 
    end
    
%    if norm(x-x_1)/norm(x)<options.stopvarx
    if norm(x-x_1,'inf')/max(1,norm(x,'inf'))<options.stopvarx
%         loop=0;
        test = test + 1 ;
        if options.verbose
        disp('delta x convergence')
        end
    end
    
    if abs(loss(end)-loss(end-1))/abs(loss(end-1))<options.stopvarj
%         loop=0;
        test = test + 1 ;
        if options.verbose
        disp('delta cost convergence')
        end
    end
    
    if it>=options.nbitermax
        loop=0;
        if options.verbose
        disp('max number iteration reached')
        end
    end
    
    if test>=3
        loop=0;
        if options.verbose
        disp('3 criteres de cv atteints')
        end
    end
    
    it=it+1;
    
    
    
end


LOG.loss=loss ;