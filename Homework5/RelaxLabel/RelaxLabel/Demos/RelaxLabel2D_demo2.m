function RelaxLabel2D_demo2
% Segment an RGB image using bisecting k-means and then regularize the 
% result using probabilistic relaxation. I have not tried adding noise to 
% the image, but you are welcome to experiment.

% Read in a sample RGB image
try
    im_rgb=imread('fabric.png');
catch err %#ok<*NASGU>
    msg='You need Image Processing Toolbox to run this demo';
    disp(msg)
    return
end

% Convert to L*a*b* colour space 
t0=clock;
fprintf('Converting the image to L*a*b* colour space ...\n')
im_Lab=applycform(double(im_rgb)/255,makecform('srgb2lab'));

% Get preliminary segmentation using k-means. Looking at the image, you
% should be able to recognize 6 distinct colours. Hence we are going to be
% looking for 6 clusters. Note that I use my own implementaion of k-means,
% which unlike the standard MATLAB function, actually returns the same 
% high-quality result every time. 
siz=size(im_rgb);
X=reshape(im_Lab,[],3);
fprintf('Clustering ...\n')
L=MyKmeans(X,6); 
clear X

L=reshape(L,siz(1),[]);
class_lab = [255 255 0  ;... yellow 
             255 0   255;... magenta
             119 73  152;... purple
             0   255 0  ;... green
             0   0   0  ;... black
             255 0   0]/255;%red   
         
% Visualize the original image and prelim segmentation
close all
hf=figure('color','w','units','normalized');
imshow(im_rgb)
set(get(gca,'Title'),'String','ORIGINAL','FontSize',25)
set(hf,'position',[0.1 0.1 0.7 0.7]);
clear im_rgb

hf=figure('color','w','units','normalized'); 
imshow(L,class_lab)
set(get(gca,'Title'),'String','BISECTING K-MEANS (6 classes)','FontSize',25)
set(hf,'position',[0.1 0.1 0.7 0.7]);
drawnow

% Regularize the prelim segmentation
fprintf('Performing relaxation labeling ...\n')
h = msgbox('RL in progress. Plase wait...','LR2D_demo2','help'); 
L2=RelaxLabel2D(im_Lab,L,[1 1 5]);
if ishandle(h), delete(h); end

hf=figure('color','w','units','normalized');
imshow(L2,class_lab)
set(get(gca,'Title'),'String','AFTER RELAXATION (5 iterations)','FontSize',25)
set(hf,'position',[0.1 0.1 0.7 0.7]);
fprintf('Total time: %3.0f sec\n',etime(clock,t0))
fprintf('NOTE THE CHAGES IN THE BOTTOM RIGHT CORNER OF THE CLASSIFIED IMAGE BEFORE & AFTER RELAXATION\n')


