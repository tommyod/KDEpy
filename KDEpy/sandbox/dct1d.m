function data=dct1d(data)
% computes the discrete cosine transform of the column vector data
[nrows, ncols]= size(data);
% Compute weights to multiply DFT coefficients
weight = [1;2*(exp(-i*(1:nrows-1)*pi/(2*nrows))).'];
% Re-order the elements of the columns of x
data = [ data(1:2:end,:); data(end:-2:2,:) ];
% Multiply FFT by weights:
data= real(weight.* fft(data)); % fftpack.fft

% The first data poitn differs from scipy.fftpack.dct
% but the first data point does not ever seem to be used...
end