function patches = sampleIMAGES(patchsize, numpatches)
% sampleIMAGES
% Returns numpatches patches, each of size patchsize x patchsize, for
% training
SAMPLE_SIZE = 8;

load IMAGES;    % load images from disk

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, numpatches columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

% starting iterating through patches
for i = 1:numpatches
    
    %get a random image among 10 images
    idx = randi(10,1,1);
    
    %get a random 8x8 image patch from the chosen image
    pixBegin = randi(512 - SAMPLE_SIZE -1,1,1);
    pixEnd = pixBegin + SAMPLE_SIZE -1;
    
    %isolate the patch to a 8x8 matrix
    randImg = IMAGES(pixBegin:pixEnd,pixBegin:pixEnd,idx);
    %fprintf('randImg = %d\n',size(randImg));
    
    %reshape the 8x8 matrix to 1x64 matrix
    %here, I use transpose to get the column order
    currPatch = reshape(randImg',1, SAMPLE_SIZE*SAMPLE_SIZE);
    
    %group each patch into a whole dataset
    patches(:,i)=currPatch;
end







%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
