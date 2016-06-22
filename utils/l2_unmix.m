function x=l2_unmix(y,D,lambda)

H=D'*D+lambda*eye(size(D,2));
f=-y'*D;
opts1=  optimset('display','off');

x = quadprog(H,f,[],[],[],[],zeros(size(D,2),1),[],[],opts1);