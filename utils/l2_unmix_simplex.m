function x=l2_unmix(y,D,lambda)

H=D'*D+lambda*eye(size(D,2));
f=-y'*D;
A=ones(1,size(D,2));
b=1;
x = quadprog(H,f,[],[],A,b,zeros(size(D,2),1));