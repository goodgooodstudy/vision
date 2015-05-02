function relaxed=relaximage(img,niters)%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
Pobject=img;
Pbackground=1-Pobject;

P=cat(3,Pobject,Pbackground);

% get total number of nodes (pixels), and total number of labels (object &
% bacnground
nodeSize=size(A); 
labelSize=2;

pixels= nodeSize(1)*nodeSize(2); % total number of pixels

% we have 8 neighbours ex: (n1-n8), and origin is N0)
% n1 n2 n3
% n4 N0 n5
% n6 n7 n8
neighbours = 8;
hood=[0  0; 
      0  1;
      -1  0; 
      0 -1; 
      1  0; 
      1  1; 
      -1  1; 
      -1 -1; 
      1 -1];
  
avgProb=squeeze(sum(sum(P,1),2)); % average prob of label * N_pix
P=padarray(P,[1 1 0],'replicate');

R=zeros(neighbours,labelSize,labelSize);

for i=1:neighbours % neighbour index 
    
    % Shift P to the i-th positions
    Pi=circshift(P,[hood(i,1) hood(i,2) 0]);
    
    for j=1:labelSize 
        for k=1:labelSize
            
            % Compute P(j|k)/P(j)~P(j,k)/(P(k)*P(j))=Num/Den
            Num=P(:,:,j).*Pi(:,:,k);
            Num=Num(2:end-1,2:end-1,:); % crop to image size
            R(i,j,k)=pixels*sum(Num(:))/(avgProb(j)*avgProb(k));
            
        end
    end
    
end

% Constrain all coeffs in R to the range [exp(-a) exp(a)]
a=7;
R(R<exp(-a))=exp(-a);
R(R>exp(a))=exp(a);
R=log(R)/a;

% Perform relaxation
b=0; E=zeros(1,opt(3)); thr=0;

while true
    
    b=b+1; Lo=L;
    
    % Compute neighbourhood (as opposed to neighbour) compatibilities
    Q=zeros(size(P));
    for i=1:labelSize
        for j=1:neighbours
            Pj=circshift(P,[hood(j,1) hood(j,2) 0]);
            Q(:,:,i)=Q(:,:,i)+sum(bsxfun(@times,Pj,R(j,i,:)),3);
        end
    end
    
    % Update the probabilities
    P=P.*(1+Q/neighbours);
    if opt(2)>(1+eps), P=P.^opt(2); end
    P=bsxfun(@rdivide,P,sum(P,3));

    % Crop
    P=P(2:end-1,2:end-1,:);
    
    % Check for convergence
    [~,L]=max(P,[],3);
    E(i)=sum(L(:)~=Lo(:));
    
    %disp(E(i))
    if E(i)==0, thr=thr+1; end
    if b>=niters ||  thr==3
        break; 
    end
    
    % Re-pad
    P=padarray(P,[1 1 0],'replicate');

end
relaxed=L;

