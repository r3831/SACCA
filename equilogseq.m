%% Function generates a sequence which is close to uniform grid on semilog axis

function [seq,L]=equilogseq(N,numpasses)
L=zeros(1,numpasses+1);
seq=[];
for i=1:numpasses
    tseq=logseq(N*i);
    seq=sort(union(seq,tseq),'ascend');
    L(i+1)=length(seq);
end
end

function seq=logseq(N)
mseq={[12,2,18], [21,3,27], [32,4,44], [50,10,100]};
nmseq=length(mseq);
niter=1+ceil(log10(N/100));
nseq=zeros(1,nmseq);
for i=1:nmseq
    nseq(i)=length(mseq{i}(1):mseq{i}(2):mseq{i}(3));
end
tseq=sum(nseq);
seq=zeros(1,10+(tseq*niter));
seq(1:10)=1:10; 
baseidx=10;
for iter=1:niter
    fac=10^(iter-1);
    for i=1:nmseq
        seq(baseidx+1:baseidx+nseq(i))=(fac*mseq{i}(1):fac*mseq{i}(2):fac*mseq{i}(3));
        baseidx=baseidx+nseq(i);
    end
end
seq=[seq(seq<N) N];
end

