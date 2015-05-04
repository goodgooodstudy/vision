%clear workspace
clear all
close all

imgName = 'cactus.png';
iterations = 100;
% Read in the image
img=imread(imgName);

img = double(img)/255;

Pobject=img;
Pbackground=1-Pobject;

P=cat(3,Pobject,Pbackground);

L=relaximage(img, iterations);
L=L==1;
PSNR=-10*log10(mean(abs(double(img(:))-double(L(:)))));

imshow(L)
str=sprintf('iterations = %d. noise level=%-4.3f',iterations, PSNR);
set(get(gca,'title'),'String',str,'FontSize',10)
drawnow


