%%SCCA(dataname,method,k,numpasses,ITERS,RESTART,rx,ry,Kcap) runs stochastic PLS
%  The inputs are as follows
%
%  dataname is one of these strings: 'XRMB', 'SYN'
%
%  method is one of these strings: 'msg', 'meg', 'batch', 'truth'
%
%  k a positive integer - denotes the desired RANK
%
%  numpasses is a positive integer - denotes number of passes over the data
%
%  ITERS is an array of positve integers represening which random splits use only (1-1000)
%
%  RESTART is a boolean flag - set to 1 if you want to restart from a
%  previous run - set to 0 if you want to discard previous runs
%
%  rx, ry are regularization parameters
%
%  Kcap is the hard rank constraint
%
%  The output containing the population objective, empirical objective,
%  runtime, and the singular value decomposition (U,S,V) is written to a
%  MATFILE in ../PAGE/PROFILE/CCA/METHOD/DATANAME
%  The filename follows the format: 'method_progress[rank=1,pass=1,iter=1].mat'
%
% Note: If possible combine random splits in the same run as each separate
%   run loads the data and increase I/O on the cluster
%%

function SCCA(dataname,method,k,numpasses,ITERS,RESTART,rx,ry,Kcap)

addpath(genpath('../'));

%% Default is no capping
if(nargin<9)
    Kcap=intmax;
end

%% Default is over-write the previous runs
if(nargin<6)
    RESTART=0;
end

%% Default is just one run
if(nargin<5)
    ITERS=1;
end


%% Set PAGE directories
pagepath=sprintf('../PAGE/PROFILE/CCA/%s/%s/',method,dataname);
pageprefix=@(method,rank,pass,numiter)[pagepath,...
    sprintf('%s_progress[rank=%d,pass=%d,iter=%d].mat',method,rank,pass,numiter)];
if(~exist(pagepath,'dir'))     % Check if the PAGE directory is structured properly
    flag=createpath(pagepath); % If not create the desired directory structure
    if(~flag)                  % If the directory structure could not be created
        error('Could not create path for result files');% Display error message and quit
    end
end

%% Load data
if(strcmp(dataname,'SYN'))
    load('../DATA/SYN.mat','data');
    load('../DATA/permSYN.mat','perm');
elseif(strcmp(dataname,'XRMB'))
    load('../DATA/XRMB.mat','data');
    load('../DATA/permXRMB.mat','perm');
elseif(strcmp(dataname,'mnist'))
    load('../DATA/mvmnist.mat','data');
    load('../DATA/permmnist.mat','perm');
elseif(strcmp(dataname,'mmill'))
    load('../DATA/mmill.mat','data');
    load('../DATA/permmmill.mat','perm');
elseif(strcmp(dataname,'JW11'))
    load('../DATA/JW11.mat','data');
    load('../DATA/permJW11.mat','perm');
end

M1=size(data.view1.training,1); %#ok<NODEF>
M2=size(data.view2.training,1);
d=min(M1,M2);
N=size(data.view1.training,2);

aaa=100;

for ITER=ITERS
    %% Display the run
    fprintf('Starting run: (%s,%s,%s,%d,%d)\n',...
        dataname,'CCA',method,k,ITER);
    
    %% Get a random permutation
    %     rng(ITER);
    %     iperm=randperm(2*N);
    iperm=perm(ITER,:); %#ok<NODEF>
    
    %% Random permutation of the data
    X=[data.view1.training data.view1.testing];
    data.view1.training=X(:,iperm(1:N));
    data.view1.testing=X(:,iperm(N+1:2*N));
    clear('X');
    
    Y=[data.view2.training data.view2.testing];
    data.view2.training=Y(:,iperm(1:N));
    data.view2.testing=Y(:,iperm(N+1:2*N));
    clear('Y');
    
    AutoCov=@(X) ((1/(size(X,2)))*(X-repmat(mean(X,2),1,size(X,2)))*...
        (X-repmat(mean(X,2),1,size(X,2)))');
    CrossCov=@(X,Y) ((1/(size(X,2)))*(X-repmat(mean(X,2),1,size(X,2)))*...
        (Y-repmat(mean(Y,2),1,size(X,2)))');
    
    %% Compute covariance matrix for the training and the held-out set
    
    CXY=CrossCov(data.view1.testing,data.view2.testing);
    CX=AutoCov(data.view1.testing);
    CY=AutoCov(data.view2.testing);
    [UX,SX]=eig(CX+rx*eye(size(CX)));
    [UY,SY]=eig(CY+ry*eye(size(CY)));
    T=UX*diag(diag(SX).^(-.5))*UX'*CXY*UY*diag(diag(SY).^(-.5))*UY';
    
    CXYhat=CrossCov(data.view1.training,data.view2.training);
    CXhat=AutoCov(data.view1.training);
    CYhat=AutoCov(data.view2.training);
    [UXhat,SXhat]=eig(CXhat+rx*(eye(size(CXhat))));
    [UYhat,SYhat]=eig(CYhat+ry*(eye(size(CYhat))));
    That=UXhat*diag(diag(SXhat).^(-.5))*UXhat'*CXYhat*...
        UYhat*diag(diag(SYhat).^(-.5))*UYhat';
    
    %% Compute the maximum covariance that can be captured by any k dimensions
    if(strcmp(method,'truth'))
        dS=svd(T); %#ok<NASGU>
        dS = sort(dS,'descend');
        fname=[pagepath,sprintf('truth[rank=%d,iter=%d].mat',k,ITER)];
        save(fname,'dS');
        
        dS=svd(That); %#ok<NASGU>
        dS = sort(dS,'descend');
        fname=[pagepath,sprintf('truth_emp[rank=%d,iter=%d].mat',k,ITER)];
        save(fname,'dS');
        continue;
    end
    
    %% Sequence of iterations on which to compute objective
    [seq,L]=equilogseq(N,numpasses);
    offset=1;
    
    %% Initialize the basis for MSG and MEG
    S=0;
    if(strcmp(method,'meg'))
        U=orth(random('normal',0,1,M1+M2,M1+M2));
        V=[]; % not needed
        %         V=orth(random('normal',0,1,M2+M1,M1+M2));
        %W = orth(random('normal',0,1,M1+M2,k));
        %maybe a different init
        S=ones(M1+M2,1)/(M1+M2);
        %M = eye(M1+M2,M1+M2)/(M1+M2);
        UX=zeros(M1,1); SX=0;
        UY=zeros(M2,1); SY=0;
    elseif (strcmp(method,'msg'))
        U=orth(random('normal',0,1,M1,k));
        V=orth(random('normal',0,1,M2,k));
        %         S=1;
        %maybe a different init
        S=ones(k,1);
        UX=zeros(M1,1); SX=0;
        UY=zeros(M2,1); SY=0;
    end
    
    %% Initialize objective value and runtime
    objV=zeros(L(numpasses+1),1);
    objVe=zeros(L(numpasses+1),1);
    runtime=zeros(L(numpasses+1),1);
    iterrank=zeros(L(numpasses+1),1);
    
    %% Check if we can start from a previous run
    firstpass=1;
    initsamp=1;
    if(RESTART)
        for ipass=numpasses:-1:0
            if(exist(pageprefix(method,k,ipass,ITER),'file'))
                fprintf('Loading state from a previous run at pass number: %d\n',ipass);
                load(pageprefix(method,k,ipass,ITER),'runtime',...
                    'U','S','V','objV','objVe','seq');
                firstpass=ipass;
                if(isempty(find(objV>0,1,'last')))
                    offset=1;
                else
                    offset=find(objV>0,1,'last');
                end
                if(offset>=L(ipass+1))
                    firstpass=firstpass+1;
                    offset=1;
                end
                break;
            end
        end
    end
    
    %% Loop over the passes
    for ipass=firstpass:numpasses % ipass is the number of current pass over data
        
        fprintf('Starting pass number: %d\n',ipass);
        
        %% Output filename
        fname=pageprefix(method,k,ipass,ITER);
        
        %% Loop over data
        for iter=L(ipass)+offset:L(ipass+1)
            fprintf('Sequence number %d...',seq(iter));
            switch(method)
                case 'batch'
                    %% BATCH CCA
                    isamp=seq(iter);
                    if(isamp<=10)
                        fprintf('\n');
                        continue;
                    end
                    if ipass>1
                        isamp=N;
                    end
                    tcounter=tic;
                    CXYhat=CrossCov(data.view1.training(:,1:isamp),data.view2.training(:,1:isamp));
                    CXhat=AutoCov(data.view1.training(:,1:isamp));
                    CYhat=AutoCov(data.view2.training(:,1:isamp));
                    [UXhat,SXhat]=eig(CXhat+rx*eye(size(CXhat)));
                    [UYhat,SYhat]=eig(CYhat+ry*eye(size(CYhat)));
                    That=UXhat*diag(diag(SXhat).^(-.5))*UXhat'*CXYhat*...
                        UYhat*diag(diag(SYhat).^(-.5))*UYhat';
                    [U,S,V]=svd(That,0);
                    [dS, idx]= sort(diag(S), 'descend');
                    S = diag(dS);
                    U = U(:, idx);
                    V = V(:, idx);
                    runtime(iter)=toc(tcounter);
                    k2=min(size(U,2),size(V,2));
                    k3=min(k,k2);
                    S=diag(S(1:k3,1:k3));
                    U=U(:,1:k3);
                    V=V(:,1:k3);
                    
                case 'ccalin'
                    %% CCALIN
                    isamp=seq(iter);
                    if(isamp<=aaa)
                        fprintf('\n');
                        continue;
                    end
                    if ipass>1
                        isamp=N;
                    end
                    tcounter=tic;
                    [ U,V,~ ] = CCALin(data.view1.training(:,1:isamp), ...
                        data.view2.training(:,1:isamp),rx,ry,1,6,k );
                    runtime(iter)=toc(tcounter);
                    %disp(U'*data.view1.training(:,1:isamp)*data.view1.training(:,1:isamp)'/N*U)
                    
                case 'ccaals'
                    isamp=seq(iter);
                    if(isamp<=aaa)
                        fprintf('\n');
                        continue;
                    end
                    if ipass>1
                        isamp=N;
                    end
                    tcounter=tic;
                    %                     [ U,V,~] = ALS_CCA(data.view1.training(:,1:isamp), ...
                    %                         data.view2.training(:,1:isamp),rx,ry,1,5,k);
                    [U,V,~,~] = CCA_als_svrg_reg(data.view1.training(:,1:isamp)', ...
                        data.view2.training(:,1:isamp)',rx);
                    runtime(iter)=toc(tcounter);
                    
                case 'msg'
                    %% MSG-CCA
%                     Kcap = 100;
                    for isamp=initsamp:seq(iter)
                        modisamp=1+mod(isamp-1,N);
                        if(isamp==1)
                            tcounter=tic;
                            SX=norm(data.view1.training(:,1));
                            UX=data.view1.training(:,1)/SX;
                            SY=norm(data.view2.training(:,1));
                            UY=data.view2.training(:,1)/SY;
                            runtime(iter)=runtime(iter)+toc(tcounter);
                            continue;
                        end
                        tcounter=tic;
                        [UX,SX]=updateEig_Cx(UX,SX,isamp,...
                            data.view1.training(:,modisamp), 3*Kcap);
                        [UY,SY]=updateEig_Cx(UY,SY,isamp,...
                            data.view2.training(:,modisamp), 3*Kcap);
                        if(isamp<=aaa)
                            runtime(iter)=runtime(iter)+toc(tcounter);
                            continue;
                        end
                        eta_t=.1/sqrt(isamp - aaa+1);
                        wx=UX*diag(1./sqrt(SX+rx)-1/sqrt(rx)) * ...
                            (UX'*data.view1.training(:,modisamp)) + ...
                            1/sqrt(rx)*data.view1.training(:,modisamp);
                        wy=UY*diag(1./sqrt(SY+ry)-1/sqrt(ry)) * ...
                            (UY'*data.view2.training(:,modisamp)) + ...
                            1/sqrt(ry)*data.view2.training(:,modisamp);
                        [U,S,V]=updateSVD_Cxy(U,S,V,...
                            eta_t*wx, wy, max(M1,M2));
                        
                        % S is sorted in ascending order
                        [S,idx] = sort(S,'ascend');
                        U=U(:,idx);
                        V=V(:,idx);
                        S=the_projection( S, k );
                        
                        
                        [S,idx] = sort(S,'descend');
                        S=S(1:min(Kcap,length(S)));
                        U=U(:,idx(1:min(Kcap,length(S))));
                        V=V(:,idx(1:min(Kcap,length(S))));
                        
                        runtime(iter)=runtime(iter)+toc(tcounter);
                    end
                    k2=min(size(U,2),size(V,2));
                    S=S(1:k2);
                    U=U(:,1:k2);
                    V=V(:,1:k2);
                    
                case 'meg'
                    %% MEG-CCA
                    for isamp=initsamp:seq(iter)
                        modisamp=1+mod(isamp-1,N);
                        if(isamp==1)
                            tcounter=tic;
                            SX=norm(data.view1.training(:,1));
                            UX=data.view1.training(:,1)/SX;
                            SY=norm(data.view2.training(:,1));
                            UY=data.view2.training(:,1)/SY;
                            runtime(iter)=runtime(iter)+toc(tcounter);
                            continue;
                        end
                        tcounter=tic;
                        if(isamp<=aaa)
                            [UX,SX]=updateEig_Cx(UX,SX,isamp,...
                                data.view1.training(:,modisamp), M1);
                            [UY,SY]=updateEig_Cx(UY,SY,isamp,...
                                data.view2.training(:,modisamp), M2);
                            runtime(iter)=runtime(iter)+toc(tcounter);
                            continue;
                        end                        
                        [UX,SX]=updateEig_Cx(UX,SX,isamp,...
                            data.view1.training(:,modisamp), 25);
                        [UY,SY]=updateEig_Cx(UY,SY,isamp,...
                            data.view2.training(:,modisamp), 25);
                        
                        eta_t=.1/sqrt(isamp - aaa+1);
%                         eta_t = .1/(sqrt((ipass-1)*N+isamp));                        
                        wx=UX*diag(1./sqrt(SX+rx)-1/sqrt(rx)) * ...
                            (UX'*data.view1.training(:,modisamp)) + ...
                            1/sqrt(rx)*data.view1.training(:,modisamp);
                        wy=UY*diag(1./sqrt(SY+ry)-1/sqrt(ry)) * ...
                            (UY'*data.view2.training(:,modisamp)) + ...
                            1/sqrt(ry)*data.view2.training(:,modisamp);
                        [U,S]=meg_pls(k,U,S,wx,wy,eta_t,max(M1,M2));
                        runtime(iter)=runtime(iter)+toc(tcounter);
                    end
            end
            initsamp=seq(iter)+1;
            keff=min(k,size(U,2));
            if(sum(strcmp(method,{'msg'}))) %JUST LIKE WARMUTH, NOT LIKE MSG PCA
                [U_msg,V_msg]=msg_pls_solution(keff,U,S,V);
                objV(iter)=trace(abs(U_msg(:,1:keff)'*T*V_msg(:,1:keff)));
                objVe(iter)=trace(abs(U_msg(:,1:keff)'*That*V_msg(:,1:keff)));
                iterrank(iter)=length(S);
            elseif(strcmp(method,{'meg'}))
                %[U,S] = eig((M+M')/2);
                keff = min(k,length(S));
                [U_meg,V_meg]=warmuth_pls_solution(keff,U,diag(S),U);
                %                 [U_meg,V_meg]=warmuth_pls_solution2(keff,U,S,M1);
                objV(iter)=trace(abs(U_meg(:,1:keff)'*[zeros(size(T,1)),T ; T',zeros(size(T,2))]*V_meg(:,1:keff)));
                objVe(iter)=trace(abs(U_meg(:,1:keff)'*[zeros(size(That,1)), That; That', zeros(size(That,2))]*V_meg(:,1:keff)));
                %                 objV(iter)=trace(abs(U_meg(:,1:keff)'*T*V_meg(:,1:keff)));
                %                 objVe(iter)=trace(abs(U_meg(:,1:keff)'*That*V_meg(:,1:keff)));
                %                 disp(objV(iter))
                %                 disp(objVe(iter))
                iterrank(iter)=length(S);
                %                 objVe(iter) = trace(abs(U_meg*U_meg'*[zeros(size(That,1)),That;That',zeros(size(That,2))]));
            elseif(any(strcmp(method,{'ccalin','ccaals'})))
                %disp(eye(size(k))-abs(U'*(CX+rx*eye(size(CX)))*U));
                %disp(V'*(CY+ry*eye(size(CY)))*V);
                objV(iter)=trace(abs(U(:,1:keff)'*CXY*V(:,1:keff)));
                %                 disp(objV(iter));
                objVe(iter)=trace(abs(U(:,1:keff)'*CXYhat*V(:,1:keff)));
                %                 disp(objVe(iter));
                iterrank(iter)=keff;
            else
                objV(iter)=trace(abs(U(:,1:keff)'*T*V(:,1:keff)));
                objVe(iter)=trace(abs(U(:,1:keff)'*That*V(:,1:keff)));
                iterrank(iter)=keff;
            end
            save(fname,'runtime','U','S','V','objV','objVe','iterrank','seq');
            fprintf('objV: %f, objVe: %f, rank: %d\n',objV(iter),objVe(iter),iterrank(iter));
        end
        save(fname,'runtime','U','S','V','objV','objVe','iterrank','seq');
    end
end
end
