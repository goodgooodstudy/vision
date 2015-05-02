%clear workspace
clear all
close all

imgName = 'cactus.png';

% Read in the image
img=imread(imgName);
%display the input image
figure('color','w','units','normalized','position',[0.1 0.1 0.8 0.8])
subplot(2,2,1)
imshow(img)
set(get(gca,'title'),'String','ORIGINAL','FontSize',20)

img = double(img)/255;

Pobject=img;
Pbackground=1-Pobject;

P=cat(3,Pobject,Pbackground);

L=RelaxLabel2D(P,[],[2 1 150]);
L=L==1;
PSNR=-10*log10(mean(abs(double(img(:))-double(L(:)))));

subplot(2,2,2)
imshow(L)
str=sprintf('RL 30 iter. PSNR=%-4.3f',PSNR);
set(get(gca,'title'),'String',str,'FontSize',20)
drawnow


