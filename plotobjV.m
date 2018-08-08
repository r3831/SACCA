%%PLOTOBJV(dataname,methods,PCA,k,numpasses,ITER,AVG,PLOTREPORT) plots
% PROGRESS-PER-ITERATION (empirical & population) and PROGRESS-PER-SECOND.
%
%  The inputs are as follows:
%
%  dataname is one of these strings: 'XRMB', 'JW11', 'SYN'
%  method is a cell of strings: e.g. {'msg', 'meg', 'batch', 'truth'}
%
%  k a positive integer - denotes the desired RANK
%
%  numpasses is a positive integer - denotes number of passes over the data
%
%  ITERS is an array of positve integers represening which random splits use only (1-1000)
%
%  AVG is either an empty string '' in which all iterations are plotted
%  simultaneously or 'avg' in which case all iterations are avergaged
%
%  PLOTREPORT is a boolean flag - set to 0 if you want to generate a report
%  on the cluster - set to 1 if you also want to plot the report
%
%  The output containing the report is written to ../REPORT as a MAT file
%  in the following format: reportPCA[method=sgd,rank=4,numiter=10].mat
%
%  If PLOTREPORT is set to 1, three plots are generated and written to
%  ../PLOTS as pdf files with names in the following formats:
%
%  progress_iteration_pop_PCA[dataname=VidTIMIT,rank=1,numiter=1,methods=sgd,brand]
%  progress_iteration_emp_PCA[dataname=VidTIMIT,rank=1,numiter=1,methods=sgd,brand].pdf
%  progress_runtime_pop_PCA[dataname=VidTIMIT,rank=1,numiter=1,methods=sgd,brand].pdf
%
%%

function [figs,fnames]=plotobjV(dataname,methods,k,numpasses,ITER,AVG,PLOTREPORT)

%% Parse inputs
if(nargin<7)
    PLOTREPORT=0;
end
if(nargin<6)
    AVG='';
end
if(nargin<5)
    ITER=1;
end

%% Don't plot!!
REPORT=1;
if(PLOTREPORT)
    REPORT=0;
end

%% Number of total random permutations
niter=length(ITER);

%% Setup Figures
col={'k','g','b','c','m','r','y'};
marker={'ks','k^','ks','kd','kh','ko','kx'};
leg=cell(length(methods)+1,1); leg{end}='Max Objective';
if(~REPORT)
    figure(11); clf;
    figure(22); clf;
end

%% File names of figures to be plotted
fnames=cell(2,1);
fname11=sprintf(['../PLOTS/progress_iteration_pop_CCA[dataname=%s,rank=%d,',...
    'numiter=%d,methods='],dataname,k,niter);
fname22=sprintf(['../PLOTS/progress_runtime_pop_CCA[dataname=%s,rank=%d,',...
    'numiter=%d,methods='],dataname,k,niter);
for i=1:length(methods)
    fname11=sprintf('%s%s,',fname11,methods{i});
    fname22=sprintf('%s%s,',fname22,methods{i});
end
fname11=[fname11(1:end-1),'].pdf'];
fname22=[fname22(1:end-1),'].pdf'];
fnames{1}=fname11; fnames{2}=fname22;

%% Plot for each method
minruntime = 10000000;
maxruntime = 0;
for imethod=1:length(methods)
    
    method=methods{imethod};
    
    %% Method-specific flags and tags
    if(strcmp(method,'ccalin')||strcmp(method,'CCALIN'))
        method='ccalin';
        leg{imethod}='CCALin';
        methodflag(2)=1;
        icol = col{2};
        imarker = marker{2};
    elseif(strcmp(method,'meg')||strcmp(method,'MEG'))
        method='meg';
        leg{imethod}='MEG';
        methodflag(3)=1;
        icol = col{3};
        imarker = marker{3};
    elseif(strcmp(method,'msg')||strcmp(method,'MSG'))
        method='msg';
        leg{imethod}='MSG';
        methodflag(4)=1;
        icol = col{4};
        imarker = marker{4};
    elseif(strcmp(method,'cappedmsg')||strcmp(method,'CAPPEDMSG'))
        method='cappedmsg';
        leg{imethod}='CAPPED-MSG';
        methodflag(5)=1;
        icol = col{5};
        imarker = marker{5};
    elseif(strcmp(method,'ccaals')||strcmp(method,'CCAALS'))
        method='ccaals';
        leg{imethod}='ALSCCA';
        methodflag(6)=1;
        icol = col{6};
        imarker = marker{6};
    elseif(strcmp(method,'batch')||strcmp(method,'BATCH'))
        method='batch';
        leg{imethod}='batch';
        methodflag(7)=1;
        icol = col{7};
        imarker = marker{7};
    end
    
    reportfile=sprintf('../REPORT/reportCCA[method=%s,rank=%d,numiter=%d].mat',...
        method,k,niter);
    
    %% Fetch data if plotting from a report file
    if(PLOTREPORT)
        load(reportfile,'seq','runtimetotal','progressmade','progressmade_emp');
    else
        %% Set PAGE path and PAGE prefix
        if(strcmp(method,'cappedmsg'))
            mmethod='msg';
        else
            mmethod=method;
        end
        pagepath=sprintf(['../PAGE/PROFILE', ...
            '/CCA/%s/%s/'],method,dataname);
        pageprefix=@(method,rank,pass,iter)[pagepath,...
            sprintf('%s_progress[rank=%d,pass=%d,iter=%d].mat', ...
            mmethod,rank,pass,iter)];
        
        %% Set sequence points based on dataset and datasize
        if(strcmp(dataname,'SYN'))
            %N=1000;
            N = 5000;
            [seq,L]=equilogseq(N,numpasses);
        elseif(strcmp(dataname,'XRMB'))
            N=112975;
            [seq,L]=equilogseq(N,numpasses);
        elseif(strcmp(dataname,'mnist'))
            N=20000;
            [seq,L]=equilogseq(N,numpasses);
        elseif(strcmp(dataname,'mmill'))
            N=10000;
            [seq,L]=equilogseq(N,numpasses);
        elseif(strcmp(dataname,'JW11'))
            N=12000;
            [seq,L]=equilogseq(N,numpasses);
        end
        
        %% Initialize performance metrics
        if(strcmp(AVG,'avg'))
            progressmade=zeros(L(numpasses+1),1);
            progressmade_emp=zeros(L(numpasses+1),1);
            runtimetotal=zeros(L(numpasses+1),1);
        else
            progressmade=zeros(L(numpasses+1),length(ITER));
            progressmade_emp=zeros(L(numpasses+1),length(ITER));
            runtimetotal=zeros(L(numpasses+1),length(ITER));
        end
        
        %% Gather performance metrics
        citer=0;
        for iter=ITER
            citer=citer+1;
            load(pageprefix(mmethod,k,numpasses,iter),'objV','objVe','runtime');
            if(strcmp(AVG,'avg'))
                runtimetotal=runtimetotal+runtime(1:L(numpasses+1));
                progressmade=progressmade+objV(1:L(numpasses+1));
                progressmade_emp=progressmade_emp+objVe(1:L(numpasses+1));
            else
                runtimetotal(:,citer)=runtime(1:L(numpasses+1));
                progressmade(:,citer)=objV(1:L(numpasses+1));
                progressmade_emp(:,citer)=objVe(1:L(numpasses+1));
            end
        end
        
        %% Average if needed
        if(strcmp(AVG,'avg'))
            runtimetotal=runtimetotal./length(ITER);
            progressmade=progressmade./length(ITER);
            progressmade_emp=progressmade_emp./length(ITER); %#ok
        end
        
        %% Cumulate runtime for incremental/stochastic methods
        runtimetotal=cumsum(runtimetotal);
    end
    
    runtimetotal(1:10)=[];
    progressmade(1:10)=[];
    seq(1:10)=[];
    
    %% Write to a mat file if just reporting
    if(REPORT)
        fig11=-1; fig22=-1;
        save(reportfile,'seq','runtimetotal','progressmade','progressmade_emp');
    else
        fig11=figure(11);
        ignore = semilogx(seq,progressmade,icol,'LineWidth',4);
        hold on;
        if ( ~strcmp(imarker,'') )
            set( get( get( ignore, 'Annotation' ), 'LegendInformation' ), 'IconDisplayStyle', 'off' );
            subseq = round( (1:20) * length(seq) / 20 );
            semilogx(seq(subseq),progressmade(subseq),imarker,'MarkerFaceColor',icol,'MarkerSize',12);
            hold on;
        end
        fig22=figure(22);
        ignore = semilogx(runtimetotal,progressmade,icol,'LineWidth',4);
        hold on;
        if ( ~strcmp(imarker,'') )
            set( get( get( ignore, 'Annotation' ), 'LegendInformation' ), 'IconDisplayStyle', 'off' );
            subseq = round( (1:20) * length(runtimetotal) / 20 );
            semilogx(runtimetotal(subseq),progressmade(subseq),imarker,'MarkerFaceColor',icol,'MarkerSize',12);
            hold on;
        end
    end
    
    minruntime = min( minruntime, runtimetotal( 1 ) );
    maxruntime = max( maxruntime, runtimetotal( end ) );
end

%% PLOT TRUTH (if reading from a report then nothing to do)
method='truth';
if(REPORT||PLOTREPORT)
    reportfile=sprintf('../REPORT/reportCCA[method=%s,rank=%d,numiter=%d].mat',...
        method,k,niter);
end

if(PLOTREPORT)
    load(reportfile,'trueobjV','trueobjV_emp');
else
    pagepathT=sprintf('../PAGE/PROFILE/CCA/truth/%s/',dataname);
    pageprefixT1=@(rank,iter)[pagepathT,...
        sprintf('truth[rank=%d,iter=%d].mat',k,iter)];
    pageprefixT2=@(rank,iter)[pagepathT,...
        sprintf('truth_emp[rank=%d,iter=%d].mat',k,iter)];
    trueobjV=0;
    trueobjV_emp=0;
    for iter=ITER
        load(pageprefixT1(k,iter),'dS');
        trueobjV=trueobjV+sum(dS(1:k));
        load(pageprefixT2(k,iter),'dS');
        trueobjV_emp=trueobjV_emp+sum(dS(1:k));
    end
    trueobjV=(trueobjV/length(ITER))*ones(L(numpasses+1),1);
end

trueobjV(1:10)=[];


%% Plot truth if not reporting else write to a report
FSIZE1=17;
if(REPORT)
    save(reportfile,'trueobjV','trueobjV_emp');
else
    figure(11); semilogx(seq,trueobjV,col{1},'LineWidth',4); hold on;
    grid; xlabel('Iteration','FontSize',FSIZE1);%,'Interpreter','Latex');
    axis([0 seq(end) 0 max(trueobjV(1),progressmade(end))*(1.01)]);
    ylabel('Objective','FontSize',FSIZE1);%,'Interpreter','Latex');
    set(gca,'FontSize',FSIZE1);
    legend(leg,'Location','SouthEast'); %,'Interpreter','Latex'); %#ok<FNDSB>
    
    figure(22); semilogx([minruntime,maxruntime],trueobjV(1:2),col{1},'LineWidth',4); hold on;
    grid; xlabel('Runtime (in seconds)','FontSize',FSIZE1);%,'Interpreter','Latex');
    axis([0 maxruntime 0 max(trueobjV(1),progressmade(end))*(1.01)]);
    ylabel('Objective','FontSize',FSIZE1);%,'Interpreter','Latex');
    set(gca,'FontSize',FSIZE1);
end

%% Create PDFs or the report
if(~REPORT)
    topdf(fig11,fname11);
    topdf(fig22,fname22);
end

figs=[fig11 fig22];

end

