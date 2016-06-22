function [g,prox_g]=get_reg_prox(reg,params)
% return reg function and corresponding proximal operator
%
%  the regularization functions are of the form:
%       g(x)=\sum_k h(x_k)
%
%  reg: regularization term
%   - 'l2' : squared l2 ,orme (ridge regularization)
%           h(u)=u^2
%   - '0' : no reglarization 
%           h(u)=0
%   - 'set' : indicator function of a set C=[C(1) C(2)]
%           h(u)= 0 if C(1)<= u <= C(2), Inf otherwise
%           params.C : 1D set (default=[0 1]) 
%   - 'l1' : lasso l1 regularization
%           h(u)=|u|
%   - 'l1l2' : group lasso on the columns of x
%           h(u)=||u||_2
%   - 'lsp' : log sum penalty 
%           h(u)=log(1+|u|/theta)
%           params.theta: lsp param (default=1)
%   - 'mcp' : minimax concave penalty 
%           Zhang, C. H. (2010). Nearly unbiased variable selection under minimax concave penalty. The Annals of statistics, 894-942.
%   - 'lp' : lp pseudo-norm (implemented only for p=1/2)
%           h(u)=|u|^p
%           params.p: (default=1/2)
%   - 'l0' : l0 pseudo norm
%           h(u)=0 if u=0 1 otherwise
%   - 'simplex' : simplex indicator function (for projected gradient)

if nargin<2
    param=struct;
end

params=initoptions(mfilename,params,'params');

switch reg
    
    case 'l2'
        
        g=@reg_l2;
        prox_g=@prox_l2;
        
    case '0'
        
        g=@(x) 0;
        prox_g=@(x,lambda) x;        
        

     case 'set'
        
        g=@(x) reg_set(x,params.C);
        prox_g=@(x,lambda) prox_set(x,lambda,params.C);      
        
    case 'l1'
        
        g=@reg_l1;
        prox_g=@prox_l1;     

    case 'l1l2'
        
        g=@reg_l1l2;
        prox_g=@prox_l1l2;           
        
        
    case 'lsp'
        
        g=@(x) reg_lsp(x,params.theta);
        prox_g=@(x,lambda) prox_lsp(x,lambda,params.theta);
        
    case 'mcp'
        
        g=@(x) reg_mcp(x,1,params.theta);
        prox_g=@(x,lambda) prox_mcp(x,lambda,params.theta);

    case 'lp'
        
        g=@(x) reg_lp(x,params.p);
        prox_g=@(x,lambda) prox_lp(x,lambda,params.p);
        
     case 'l0'
        
        g=@(x) reg_l0(x);
        prox_g=@(x,lambda) prox_l0(x,lambda);  
        
     case 'simplex'
        
        g=@(x) reg_simplex(x);
        prox_g=@(x,lambda) prox_simplex(x,lambda);  
                
        
    otherwise
        
        error('unknown reg term')
    
end



end


function res=reg_set(x,C)
    res=sum((x<C(1))+(x>C(2)));
    if res>0
        res=1e3;
    end
end

function res=prox_set(x,lambda,C)
    res=max(min(x,C(2)),C(1));
end



function res=reg_l2(x)
    res=norm(x(1:end,:),'fro')^2/2;
end

function res=prox_l2(x,lambda)
    res=x/(1+lambda);
end

function res=reg_l1(x)
    res=sum(sum(abs(x(1:end,:))));
end

function res=prox_l1(x,lambda)
    res=sign(x).*max(abs(x)-lambda,0);
end

function res=reg_l1l2(x)
      res=sum(sqrt(sum(abs(x(1:end-1,:)).^2,2)));
end

function res=prox_l1l2(x,lambda)
res=x;
for i=1:size(x,1)
    res(i,:)=x(i,:).*max(0,1-lambda/norm(x(i,:)));
end
end

function res=reg_lsp(x,theta)
      res=sum(sum(log(1+abs(x(1:end,:))/theta)));
end

function res=prox_lsp(x,lambda,theta)
        z = abs(x) - theta;
		v = z.*z - 4.0*(lambda - abs(x)*theta);
        
        
        sqrtv = sqrt(v);
		xtemp1 = max(0,0.5*(z + sqrtv));
		xtemp2 = max(0,0.5*(z - sqrtv));

		ytemp0 = 0.5*x.*x;
		ytemp1= 0.5*(xtemp1 - abs(x)).*(xtemp1 - abs(x)) + lambda*log(1.0 + xtemp1/theta);
		ytemp2 = 0.5*(xtemp2 - abs(x)).*(xtemp2 - abs(x)) + lambda*log(1.0 + xtemp2/theta);
        
        sel1=(ytemp1<ytemp2).*(ytemp1<ytemp0);
        sel2=(ytemp2<ytemp1).*(ytemp2<ytemp0);
        
        xtemp=sel1.*xtemp1+sel2.*xtemp2;
        
        res=sign(x).*(v>0).*xtemp;
end

function res=reg_mcp(x,lambda,theta)
    indUnb = (abs(x(1:end-1,:))>theta*lambda) ;
    indBia = (abs(x(1:end-1,:))<=theta*lambda).*(abs(x(1:end-1,:))>0) ;
    res=sum(sum((x(1:end,:)-x(1:end,:).^2/(2*theta*lambda)).*indBia + theta*lambda/2*indUnb));
end

function res=prox_mcp(x,lambda,theta)
    res=zeros(size(x));
    indUnb = (abs(x(1:end,:))>theta*lambda) ;
    indBia = (abs(x(1:end,:))<=theta*lambda).*(abs(x(1:end,:))>0) ;
    res(1:end,:)=x(1:end,:).*indUnb + theta/(theta-1)*(x(1:end,:)-lambda*sign(x(1:end-1,:))).*indBia;
end

function res=reg_lp(x,p)
      res=sum(sum(abs(x(1:end,:)).^p));
end

function res=prox_lp(x,lambda,p)
res=zeros(size(x));
switch p
    case .5
        ind = (abs(x)>(.75*lambda^(2/3))) ; 
        res(ind) = 2/3*x(ind).*(1+cos(2*pi/3-2/3*acos(lambda/8*(abs(x(ind))/3).^(-3/2)))) ;
end
end

function res=reg_l0(x)
      res=sum(sum(abs(x(1:end,:)))>0);
end

function res=prox_l0(x,lambda)
thr=sqrt(2*lambda);
res=x.*(abs(x)>thr);
end


function res=reg_simplex(x)
    res=0;
end

function res=prox_simplex(x,lambda)
    res=projectSimplex(x(1:end-1));
    res(end+1)=0;
end


function [w] = projectSimplex(v)
% Computest the minimum L2-distance projection of vector v onto the probability simplex
nVars = length(v);
mu = sort(v,'descend');
sm = 0;
for j = 1:nVars
    sm = sm+mu(j);
   if mu(j) - (1/j)*(sm-1) > 0
       row = j;
       sm_row = sm;
   end
end
theta = (1/row)*(sm_row-1);
w = max(v-theta,0);
end
