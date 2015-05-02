function [IDX,C,CovMat,Dmat,Co]=MyKmeans(X,k,opt)
% Partition the data into k clusters using k-means algorithm. This function
% allows three types initializations (see below).
%
% INPUT ARGUMENTS:
%   - X     : N-by-D array of samples drawn from a continuous distribution,
%             where N = number of observations, D = dimensionality of the 
%             data.
%   - k     : this variable can be speficied in one of two ways depending 
%             on how you wish to initialize the search.
%             a) AUTOMATIC INITIALIZATION
%                  -   k is a positive integer indicating the number of 
%                      clusters. See opt below on how to specify the type 
%                      of initialization that will be used. The two choices
%                      are 'bisecting' and 'random'.
%             b) MANUAL INITIALIZATION 
%                 -    k is a K-by-D matrix, where i-th row corresponds to 
%                      the centroid of the i-th cluster.
%   - opt   : opt=[TOL IP IT] where 
%              - TOL : convergence tolerance (1E-6 is default). TOL is 
%                      measured as maximum absolute change in position of
%                      the centroids.
%              - IP  : parameter used to set the maximum number of 
%                      iterations (NMAX). NMAX=IP*K*100, where K is the
%                      number of clusters. IP=1 is default.
%              - IT  : set IT=1 to use bisecting initialization (default)
%                      and IT=2 to use random initialization.
%            
% OUTPUT:    
%   - IDX   : N-by-1 array of cluster indices assigned to points in X.
%   - C     : K-by-D array of cluster centroids.
%   - CovMat: D-by-D-by-K array of cluster covariance matrices. 
%   - Dmat  : N-by-K array of squared Euclidean distances.
%   - Co    : K-by-D array of cluster centroids used to initialize the
%             search.
%            
% REFERENES:
% [1] Bishop, C.M. Pattern Recognition and Machine Learning, Springer, 2006
% [2] Savaresi, S., Boley, D. (2001) 'On the performance of bisecting 
%     K-means and PDDP', In Proceedings of the First SIAM International
%     Conference on Data Mining (SDM’2001)
%
% AUTHOR    : Anton Semechko (a.semechko@gmail.com)
% DATE      : Nov.2011
%

% Verify the consistency of the input arguments
if nargin<2, error('Insufficient number of input arguments.'); end

if numel(k)==1 && isnumeric(k)
    K=k; % number of clusters
    C=[];
elseif ismatrix(k)
    K=size(k,1); % number of clusters
    C=k;
else
    error('Incorrect entry for 2nd input argument')
end

if nargin<3, opt=[1E-6 K*100 1]; end
CheckInputArgs(X,k,opt)

N=size(X,1); % # of data points 

% One cluster specified ---------------------------------------------------
if K==1
    IDX=ones(N,1);
    C=mean(X); Co=C;
    X=bsxfun(@minus,X,C);
    CovMat=(X'*X)/N;
    Dmat=sqrt(sum(X.^2,2));
    return
end

% More than one cluster ---------------------------------------------------

% Initialize cluster centroids
if isempty(C)
    if opt(3)==1 % bisecting initialization
        C=BisectingInitialization(X,K,opt);
    else % random initialization
        C=randperm(N);
        C=X(C(1:K),:);
    end
end
Co=C;
C=permute(C,[3 2 1]);

% Iteratively update the centroids
i=0; dC=Inf; tol=opt(1); Nmax=opt(2)*K*100; IDX=[];
while dC>tol && i<Nmax
    
    i=i+1;    
    C_old=C; 
    
    % Distance to centroids
    Dmat=sum(bsxfun(@minus,X,C).^2,2);
    
    % Select min distances and recompute the centroids 
    [~,IDX]=min(Dmat,[],3);
    for j=1:K
        C(:,:,j)=mean(X(IDX==j,:));
    end
    
    % Check for convergence
    dC=max(abs(C(:)-C_old(:)));

end
C=permute(C,[3 2 1]);

% Compute the covariance matrices
CovMat=repmat(eye(size(X,2)),[1 1 K]);
if nargout>2
    for i=1:K
        x=X(IDX==i,:);
        x=bsxfun(@minus,x,mean(x,1));
        c=(x'*x)/size(x,1);
        CovMat(:,:,i)=c;
    end
end


%==========================================================================
function C=BisectingInitialization(X,k,opt)
% Iteratively bisect the dataset into 2^nextpow2(k) clusters and select
% the centroids of the first (pointwise) k largest clusters.

[C,X1,X2]=BisectCluster(X,opt);
if k==2, return; end

D=size(X,2); % # of dimensions

% Partition the dataset
X={X1;X2};
Np=[];

n=nextpow2(k); C=[];
for i=1:n % (i+1)=hierchy level 
    
    C=zeros(2^(i+1),D);
    Xi=cell(2^(i+1),1);
    Np=zeros(2^(i+1),1);
    
    for j=1:2^i % # of sub clusters
        
        if ~isempty(X{j})
            
            [c,X1,X2]=BisectCluster(X{j},opt);
            C(2*(j-1)+1,:)=c(1,:);
            Xi{2*(j-1)+1}=X1;
            Np(2*(j-1)+1)=size(X1,1);
            if ~isempty(X2)
                C(2*j,:)=c(2,:);
                Xi{2*j}=X2;
                Np(2*j)=size(X2,1);
            end
            
        end
        
    end
    X=Xi;
end

% Select k (pointwise) largest clusters
[~,idx]=sort(Np);
C=C(idx(1:k),:);


%==========================================================================
function [C,X1,X2]=BisectCluster(X,opt)
% Bisect a point cloud into 2 clusters
%
% - C   : 2-by-D array, each row is a centroid
% - X1  : points assigned to the C(1,:) cluster
% - X2  : points assigned to the C(2,:) cluster
%

N=size(X,1);
if N<=1
    C=X; X1=X; X2=[]; return
elseif N==2
    C=X; X1=X(1,:); X2=X(2,:); return
end

if opt(3)==1 && N>3 % my method
    
    % Get the prinpical directions and the centroid
    [P,D,C]=DataPCA(X);
    P=P(:,1)'/norm(P(:,1));
    
    % Initialize the centroids along the prinicipal axis
    ds=0.2; % must be in the range 0 to 0.5
    d=rand(1)*(1-2*ds)+ds;
    C1=C+d*D(1)*P;
    C2=C-d*D(1)*P;
    
elseif N>3 % random initialization
    
    C=mean(X,1);
    C1=ceil(rand(1)*size(X,1));
    C1=X(C1,:);
    C2=C-(C-C1);
    
else
    C1=X(1,:); C2=mean(X,1);
end

% Iteratively update the centroids 
i=0; dC=Inf; tol=opt(1); Nmax=opt(2)*200; X1=[]; X2=[];
while dC>tol && i<Nmax
    
    i=i+1;    
    C1_old=C1; C2_old=C2; 
    
    % Distance to centroids
    D1=sum(bsxfun(@minus,X,C1).^2,2);
    D2=sum(bsxfun(@minus,X,C2).^2,2);
    
    % Select min distances and recompute the centroids 
    [~,idx]=min([D1 D2],[],2);
    X1=X(idx==1,:); C1=mean(X1,1);
    X2=X(idx==2,:); C2=mean(X2,1);
    
    % Check for convergence
    dC=max(abs([C1 C2]-[C1_old C2_old]));

end
C=[C1;C2];


%==========================================================================
function [P,D,C]=DataPCA(X)
% Compute the principal directions of the sample distribution.

N=size(X,1); % # of samples
C=mean(X,1); % mean 
X=bsxfun(@minus,X,C); % center on the mean
Cmat=(X'*X)/N;
[P,D,~]=eigs(Cmat); % get the principal directions
D=diag(D);


%==========================================================================
function CheckInputArgs(X,k,opt)
% Verify the consistency of the input arguments

% Make sure there are more samples than there are dimensions
if size(X,1)<=size(X,2)
    error('The number of samples should be greater than the dimensionality of the distribution')
end

chk=isnan(X)| isinf(X);
if sum(chk(:))>0, error('Input data contains NaN or Inf elements'); end

% Check opt
if ~isvector(opt) || numel(opt)~=3 || sum(opt<0 | isnan(opt))>0 || ~(opt(3)~=1 || opt(3)~=2)
    error('Incorrect entry for 3rd input argument')
end

% Check if k is a positive integer
if numel(k)==1
    if ~isnumeric(k) || round(k)~=k || k<1 || isinf(k) || isnan(k)
        error('2nd input argument must be a positive integer or an array')
    end
    return
end

% Check the consistency of dimensions
if size(k,2)~=size(X,2)
    error('The dimensionality of specified centroids does not match the dimensionality of the data')
end
