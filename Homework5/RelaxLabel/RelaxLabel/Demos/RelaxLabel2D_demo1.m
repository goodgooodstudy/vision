%clear workspace
clear all
close all

% Regularize segmented binary image.

% Names of the sample binary images ---------------------------------------
FileNames={'butterfly-1.gif','device6-3.gif'};
im0=FileNames{round(rand(1))+1};
im0 = 'cactus.png';
% Verify that the image is in the search path
if exist(im0,'file')~=2
    msg='To run this demo first unpack the contents of RelaxLabel.zip folder into your current MATLAB directory.';
    disp(msg)
    return
end

% Read in the image
try
    im0=imread(im0);
catch err %#ok<*NASGU>
    msg='You need Image Processing Toolbox to run this demo';
    disp(msg)
    return
end
temp =im0;
%im0=double(im0)/2;

bw0 = im0;
bw1 = bw0;
% % Add Gaussian as well as salt & pepper noise to the image ----------------
% bw0=im0>0; % ground truth
% im1=imnoise(im0,'gaussian',0,0.15);
% im1=imnoise(im1,'salt & pepper',0.2);
% bw1=im1>=0.5; % noisy binary image
% 
% % Compute PSNR (peak signal to noise ratio)
% PSNR=-10*log10(mean(abs(double(bw0(:))-double(bw1(:)))));
%DC=2*sum(bw0(:)&bw1(:))/(sum(bw0(:))+sum(bw1(:))); % Dice coeff

% Visualize the original an noisy images
close all
figure('color','w','units','normalized','position',[0.1 0.1 0.8 0.8])
subplot(2,2,1)
imshow(bw0)

subplot(2,2,2)
imshow(bw1)
drawnow


%--------------------------------------------------------------------------
% Initialize a priori probabilities. There are many ways to do this. The
% one I prefer goes like this: 
% 1) Apply a low pass filter to the noisy binary image (make sure it has
%    been converted to double format). 
% 2) Use filter response from step 1 to initialize the probs using the 
%    method described in the following paper:
%    Rosenfeld, A., Smith, R.C. (1981) 'Thresholding using relaxation',
%    IEEE Transactions on Pattern Analysis and Machine Intelligence,
%    Vol. PAMI-3, pp.598-606.

im_filt=imfilter(double(bw1),fspecial('gaussian',9,3)); % feel free to play around with the filter params
idx=im_filt<=0.5; % pixels whose neighbourhood mean is less than 0.5

P_drk=zeros(size(bw1)); % prob of belonging to the background
P_lgt=zeros(size(bw1)); % prob of belonging to the foreground

P_drk(idx)=(((0.5-im_filt(idx))/0.5).^4)/2+0.5;
P_lgt(idx)=1-P_drk(idx);

P_lgt(~idx)=(((im_filt(~idx)-0.5)/0.5).^4)/2+0.5;
P_drk(~idx)=1-P_lgt(~idx);

P=cat(3,P_drk,P_lgt);
% Note the formulas above have been slightly modified compared to those
% in the reference

% Perform relaxation labeling for 30 iterations---------------------------
h = msgbox('RL in progress. Plase wait...','LR2D_demo1','help'); 
L=RelaxLabel2D(P,[],[2 1 30]);
if ishandle(h), delete(h); end

% Visualize the processed image. Note L is a label image whose pixel 
% intensities correspond to differnt classes (in this case background and 
% foreground)
L=L==2;
PSNR=-10*log10(mean(abs(double(bw0(:))-double(L(:)))));
%DC=2*sum(bw0(:)&L(:))/(sum(bw0(:))+sum(L(:)));

subplot(2,2,3)
imshow(L)
str=sprintf('RL 30 iter. PSNR=%-4.3f',PSNR);
set(get(gca,'title'),'String',str,'FontSize',20)
drawnow


% For comparison, clean up the noisy image using morphological filter -----
bw2=bwareaopen(bw1,10); % size filter
bw2=imclose(bw2,strel('disk',4));

PSNR=-10*log10(mean(abs(double(bw0(:))-double(bw2(:)))));
%DC=2*sum(bw0(:)&bw2(:))/(sum(bw0(:))+sum(bw2(:)));

subplot(2,2,4)
imshow(bw2)
str=sprintf('MORPH.FILT. PSNR=%-4.3f',PSNR);
set(get(gca,'title'),'String',str,'FontSize',20)
drawnow

