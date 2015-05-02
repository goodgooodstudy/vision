function [Lout,P,E]=RelaxLabel2D(A,L,opt)
% Improve the spatial consistency and structural coherence of a 
% presegmented/classifed 2D monochromatic or multispectral image by 
% performing nonlinear relaxation labeling (RL). 
%
% SYNTAX:
%   1) You have the original image, A, and the corresponding label image, 
%      L, then use the following syntax L=RelaxLabel2D(A,L,opt). Note that
%      in this case, the PDF of every label will be estimated using a 
%      simple, unimodal Gaussian model.
%   2) You have a M-by-N-by-K array of prior label (i.e. class) 
%      probabilities, where K is the total number of labels, then use this
%      syntax L=RelaxLabel2D(A,[],opt).
%
% INPUT ARGUMENTS:
%   - A     : this variable can be specified in one of two ways:
%             (1) A is a M-by-N monochromatic or M-by-N-by-D mutispectral 
%                 image. 
%             (2) A is M-by-N-by-K array of prior probabilities where K
%                 is the total number of labels (ie. classes). For example,
%                 A(i,j,k) is the probability that pixel in position (i,j)
%                 belongs to the k-th class. 
%   - L     : if A is specified as in (2) set L=[]. Otherwise L is a 
%             M-by-N array of preliminary labels. For binary label problems
%             L can be specified as a binary (i.e. logical) image. For 
%             multi-label problems, L contains integer labels corresponding
%             to the specific classes assigned to pixels in A. Maximum
%             allowed number of labels is 10. 
%   - opt   : opt=[flag w Nmax conn], where:
%             - flag = 1 if A is specified according to (1) as explained
%             above. flag = 2 is A is specified according to (2). The 
%             former is the default setting.
%             - w is a real number in the range [1 2] used to modify the
%             rate of convergence. w=1 is default.
%             - Nmax is the maximum number of iterations. Nmax=10 is
%             default.
%             - conn is an an integer indicating neighbourhood 
%             connectivity. Two options are conn=4 and conn=8. Although the
%             latter is the default setting, if you are working with large
%             images then setting conn to 4 will improve time performance.
%             
% OUTPUT:
%   - Lout  : new label matrix.
%   - P     : M-by-N-by-K array of class probabilities after RL where K
%             is the total number of labels.
%   - E     : 1-by-s vector, where E(i) is total number of pixels whose 
%             labels were modified during i-th iteration; s is the total 
%             number of iterations. 
%
% REFERENCES:
% [1] Eklundh, J.O., Yamamoto, H., Rosenfeld, A. (1980) 'A relaxation 
%     method for multispectral pixel classification', IEEE Transactions on
%     Pattern Analysis and Machine Intelligence, Vol. PAMI-2, pp.72-75.
% [2] Kittler, J., Illingworth, J. (1985) 'Relaxation labelling 
%     algorithms a review', Image and Vision Computing, Vol.3, pp.206-216.
% [3] Peleg, S., Rosenfeld, A. (1978) 'Determining compatibility 
%     coefficients for curve enhancement relaxation processes', IEEE 
%     Transactions on Systems, Man, and Cybernetics, Vol. SMC-8, pp.548-555.
%
% AUTHOR: Anton Semechko (a.semechko@gmail.com)
% DATE: Jan.2011
%

% Check the format of the main array
siz_A=size(A); D=size(A,3);


%============================= MAIN BODY ==================================

N_pix=siz_A(1)*siz_A(2); % total number of pixels

if opt(1)==1
    % Estimate the Gaussian parameters of every label class
    A=reshape(A,[],D);
    l=L(:);
    mu=zeros(N_lab,D); % mean vectors
    C=zeros(D,D,N_lab); %covariance matrices
    W=zeros(N_lab,1); % class probabilities
    for i=1:N_lab
        X=A(l==i,:); % pixels with labels i
        n=size(X,1);
        if n<2, error('Only one pixel detected for label #%u',i); end
        [mu(i,:),C(:,:,i)]=GaussParams(X);
        W(i)=n/size(A,1);
    end
    clear X n
    
    % Compute the membership probabilities of all pixels
    P=zeros(size(A,1),N_lab);
    for i=1:N_lab
        P(:,i)=W(i)*GaussPDF(A,mu(i,:),C(:,:,i));
    end
    P=bsxfun(@rdivide,P,sum(P,2));
    P=reshape(P,siz_A(1),siz_A(2),[]);
    clear mu C W l
else
    P=A;
end
clear A


% Compute neighbour support coefficients (aka compatibility coefficients)
%--------------------------------------------------------------------------
% Specify displacements with respect to the neighbours
N_ngb=opt(4)+1;
cs=[0  0; 0  1;-1  0; 0 -1; 1  0];
if opt(4)==8, cs=cat(1,cs,[1  1;-1  1;-1 -1; 1 -1]); end

P_lab_ave=squeeze(sum(sum(P,1),2)); % average prob of label * N_pix
P=padarray(P,[1 1 0],'replicate');

R=zeros(N_ngb,N_lab,N_lab);
for i=1:N_ngb % neighbour index 
    
    % Shift P to the i-th positions
    Pi=circshift(P,[cs(i,1) cs(i,2) 0]);
    
    for j=1:N_lab 
        for k=1:N_lab
            
            % Compute P(j|k)/P(j)~P(j,k)/(P(k)*P(j))=Num/Den
            Num=P(:,:,j).*Pi(:,:,k);
            Num=Num(2:end-1,2:end-1,:); % crop to image size
            R(i,j,k)=N_pix*sum(Num(:))/(P_lab_ave(j)*P_lab_ave(k));
            
        end
    end
    
end
clear Pi P_lab_ave Num

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
    for i=1:N_lab
        for j=1:N_ngb
            Pj=circshift(P,[cs(j,1) cs(j,2) 0]);
            Q(:,:,i)=Q(:,:,i)+sum(bsxfun(@times,Pj,R(j,i,:)),3);
        end
    end
    
    % Update the probabilities
    P=P.*(1+Q/N_ngb);
    if opt(2)>(1+eps), P=P.^opt(2); end
    P=bsxfun(@rdivide,P,sum(P,3));

    % Crop
    P=P(2:end-1,2:end-1,:);
    
    % Check for convergence
    [~,L]=max(P,[],3);
    E(i)=sum(L(:)~=Lo(:));
    
    %disp(E(i))
    if E(i)==0, thr=thr+1; end
    if b>=opt(3) ||  thr==3
        break; 
    end
    
    % Re-pad
    P=padarray(P,[1 1 0],'replicate');

end
Lout=L;
E=E(1:b);


%==========================================================================
function p=GaussPDF(X,mu,C)
% Probability of X given mu and C.

N_dim=size(X,2); % number of dims

% Center
X=double(X);
X=bsxfun(@minus,X,mu);

% Mahalanobis distance
D=sum(X.*(C\X')',2);

% Probability
p=1/sqrt(det(C))*(1/sqrt(2*pi))^N_dim;
p=p*exp(-D/2);


%==========================================================================
function [mu,C]=GaussParams(X)
% Estimate the parameters of the normal pdf.

X=double(X);

% Find the mean and center
mu=mean(X,1);
X=bsxfun(@minus,X,mu);

% Compute the covariance
n=size(X,1);
C=(X'*X)/(n-1);

