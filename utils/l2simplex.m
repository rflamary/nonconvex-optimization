function res=l2simplex(x,lambda)
    if lambda>=1
       res=zeros(size(x));
       ind=find(x==max(x));
       res(ind(1))=1;
    else
    res=projectSimplex(x/(1-lambda));
    end
end