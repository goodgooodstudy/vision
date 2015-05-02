function RelaxLabel2D_demo3
% Segment one of the sample graysale images into foreground and background.
% In my experience, I have found relaxation labeling to be most useful in 
% segmenting natural, noisy images where the boundaries of objects are 
% poorly defined. These examples of CLSM images of biofilm and bioplastic 
% are meant to demonstrate it. 

% Names of the sample images ----------------------------------------------
FileNames={'biofilm1','biofilm2','soy1','soy2'};
%im0=FileNames{ceil(4*rand(1))};
im0=FileNames{3};
im0=strcat('CLSM_',im0,'.tif');

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

% Normalize all intensities to the range [0 1] 
im0=double(im0);
im0=im0-min(im0(:));
im0=im0/max(im0(:));


% Visualize the image
close all
figure('color','w')
imshow(im0)
set(get(gca,'title'),'String','ORIGINAL IMAGE','FontSize',20)
drawnow

% Presegment the image using Otsu thrsholding
im1=im0>graythresh(im0);%*255;

figure('color','w')
imshow(im1)
set(get(gca,'title'),'String','OTSU THRESHOLDING','FontSize',20)


% Initialize a priori probabilites of pixels belonging to the foreground
% and background classes. Same idea as in the first demo.

im2=imfilter(double(im1),fspecial('gaussian',7,2)); % feel free to play around with the filter params
idx=im0<=0.5; % pixels whose neighbourhood mean is less than 0.5

P_drk=zeros(size(im0)); % "prob" of belonging to the background
P_lgt=zeros(size(im0)); % "prob" of belonging to the foreground

P_drk(idx)=(((0.5-im2(idx))/0.5).^1)/2+0.5;
P_lgt(idx)=1-P_drk(idx);

P_lgt(~idx)=(((im2(~idx)-0.5)/0.5).^1)/2+0.5;
P_drk(~idx)=1-P_lgt(~idx);

P=cat(3,P_drk,P_lgt);


% Perform probabilistic relaxation for 30 iterations to regularize the 
% segmented image
h = msgbox('RL in progress. Plase wait...','LR2D_demo3','help'); 
L=RelaxLabel2D(P,[],[2 1 30 8]);
if ishandle(h), delete(h); end

figure('color','w')
imshow(L==2)
set(get(gca,'title'),'String','AFTER RELAXATION LABELING','FontSize',20)


