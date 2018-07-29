

n = 100
X = (1:n).^1
X = lognrnd(1,1, n, 1)
X = reshape(X, [n,1])
[pdf, grid] = akde1d(X);

% plot(grid, pdf)


data = lognrnd(1,1, n, 1) + 3;
data = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6]
n = 2^10;
MIN = -5;
MAX = 12;
[bandwidth,density,xmesh,cdf]=kde(data,n,MIN,MAX)

%plot(xmesh, density)

data=[randn(100,1);randn(100,1)*2+35 ;randn(100,1)+55];
[bandwidth,density,xmesh,cdf]= kde(data,2^14,min(data)-5,max(data)+5);
plot(xmesh, density)