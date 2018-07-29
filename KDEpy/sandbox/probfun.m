function out=probfun(x,w,mu,Sig)
[gam,d]=size(mu);
out=0;
for k=1:gam
    S=Sig(k,:);
    xx=bsxfun(@minus, x,mu(k,:));
    xx=bsxfun(@rdivide,xx.^2,S);
    out=out+exp(-.5*sum(xx,2)+log(w(k))-.5*sum(log(S))-d*log(2*pi)/2);
end
end
