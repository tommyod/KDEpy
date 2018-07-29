function [w,mu,Sig,del,ent]=regEM(w,mu,Sig,del,X)
[gam,d]=size(mu);
[n,d]=size(X);
log_lh=zeros(n,gam); 
log_sig=log_lh;
for i=1:gam
    s=Sig(i,:);
    Xcentered = bsxfun(@minus, X, mu(i,:));
    xRinv = bsxfun(@rdivide, Xcentered.^2, s);
    xSig = sum(bsxfun(@rdivide, xRinv, s),2)+eps;
    log_lh(:,i)=-.5*sum(xRinv, 2)-.5*sum(log(s))+log(w(i))-d*log(2*pi)/2-.5*del^2*sum(1./s);
    log_sig(:,i)=log_lh(:,i)+log(xSig);
end
maxll = max (log_lh,[],2); 
maxlsig = max (log_sig,[],2);
p= exp(bsxfun(@minus, log_lh, maxll));
psig=exp(bsxfun(@minus, log_sig, maxlsig));
density = sum(p,2); psigd=sum(psig,2);
logpdf=log(density)+maxll; 
logpsigd=log(psigd)+maxlsig;
p = bsxfun(@rdivide, p, density);% normalize classification prob.
ent=sum(logpdf); 
w=sum(p,1);
for i=find(w>0)
    mu(i,:)=p(:,i)'*X/w(i);  %compute mu's
    Xcentered = bsxfun(@minus, X,mu(i,:));
    Sig(i,:)=p(:,i)'*(Xcentered.^2)/w(i)+del^2; % compute sigmas
end
w=w/sum(w);
curv=mean(exp(logpsigd-logpdf));
del=1/(4*n*(4*pi)^(d/2)*curv)^(1/(d+2));
end