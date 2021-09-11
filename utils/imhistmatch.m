function varargout = imhistmatch(varargin)
%IMHISTMATCH Adjust image to match its histogram to that of another image.
%   B = IMHISTMATCH(A,REF) transforms the input grayscale or truecolor
%   image A so that the histogram of the output image B approximately
%   matches the histogram of the reference image REF, when the same number
%   of bins are used for both histograms. For truecolor images, each color
%   channel of A is matched independently to the corresponding color
%   channel of REF.
%
%   B = IMHISTMATCH(A, REF, N) uses N equally spaced histogram bins for
%   transforming input image A. The image returned in B has N discrete
%   levels. The default value for N is 64.
%
%   [B, HGRAM] = IMHISTMATCH(A, REF,...) also returns the histogram of the
%   reference image REF used for matching in HGRAM. HGRAM is a 1 x N (when
%   REF is grayscale) or a 3 x N (when REF is truecolor) matrix, where N is
%   the number of histogram bins. Each row in HGRAM stores the histogram of
%   a single color channel of REF.
% 
%   Notes 
%   ----- 
%   1. The histograms for A and REF are computed with equally spaced bins 
%      and with intensity values in the appropriate range for each image:
%      [0,1] for images of class double or single, [0,255] for images of
%      class uint8, [0,65535] for images of class uint16, and [-32768,
%      32767] for images of class int16.
%
%   Class Support
%   -------------
%   A can be uint8, uint16, int16, double or single. The output image B has 
%   the same class as A. The optional output HGRAM is always of class double.
%
%   Example
%   -------
%   This example matches the histogram of one image to that of another image
%
%      A = imread('office_2.jpg');
%      imshow(A, []);
%      title('Original Image');
%
%      ref = imread('office_4.jpg');
%      figure, imshow(ref, []);
%      title('Reference Image');
%
%      B = imhistmatch(A, ref);
%
%      figure, imshow(B, []);
%      title('Histogram Matched Image');
%
%   See also HISTEQ, IMADJUST, IMHIST.

%   Copyright 2012 The MathWorks, Inc.

% The output is stored back in A to lower internal memory usage.

narginchk(2,3);
nargoutchk(0,2);

[A, ref, N] = parse_inputs(varargin{:});
% If input A is empty, then the output image B will also be empty

numColorChan = size(ref,3);

isColor = numColorChan > 1;

% Compute histogram of the reference image
hgram = zeros(numColorChan,N);
for i = 1:numColorChan
    hgram(i,:) = imhist(ref(:,:,i),N);
end

% Adjust A using reference histogram
hgramToUse = 1;
for k = 1:size(A,3) % Process one color channel at a time
    if isColor
        hgramToUse = k; % Use the k-th color channel's histogram
    end
    
    for p = 1:size(A,4)
        % Use A to store output, to save memory
        A(:,:,k,p) = histeq(A(:,:,k,p), hgram(hgramToUse,:)); 
    end    
end

% Set output arguments
varargout{1} = A;  % Always set varargout{1} so 'ans' gets populated even if user doesn't ask for output
if (nargout == 2)
    varargout{2} = hgram;
end

end

%======================================================================

function [A, ref, N] = parse_inputs(varargin)

A    = varargin{1};
validateattributes(A,{'uint8','uint16','double','int16', ...
    'single'},{'nonsparse','real'}, mfilename,'A',1);

ref = varargin{2};
validateattributes(ref,{'uint8','uint16','double','int16', ...
    'single'},{'nonsparse','real','nonempty'}, mfilename,'ref',2);

if (ndims(ref) > 3)
    error(message('images:validate:tooManyDimensions','ref',3));
end

if ((size(A,3) ~= size(ref,3)) && (size(ref,3) > 1) && ~isempty(A))
    error(message('images:validate:unequalNumberOfColorChannels','A','ref'));
end

if (nargin == 3)
    N = varargin{3};
    validateattributes(N,{'numeric'},{'scalar','nonsparse','integer', ...
        '>', 1}, mfilename,'N',3);
else    
    N = 64;
end

end




