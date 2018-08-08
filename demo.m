% dataname='JW11';
% dataname='SYN';
dataname='mmill';
% dataname='mnist';
% methods={'batch','msg','meg','ccalin','ccaals'};
methods={'batch','msg','meg','ccalin'};
% methods={'batch','msg','ccalin'};
% methods={'batch','msg'};
k=4; Kcap=k+3; numpasses=2; ITERS=1:1; RESTART=0; rx=.1; ry=.1;

% SCCA(dataname,'batch',k,numpasses,ITERS,RESTART,rx,ry,Kcap)
% SCCA(dataname,'truth',k,numpasses,ITERS,RESTART,rx,ry,Kcap)
SCCA(dataname,'msg',k,numpasses,ITERS,RESTART,rx,ry,Kcap)
% SCCA(dataname,'ccalin',k,numpasses,ITERS,RESTART,rx,ry,Kcap)
% SCCA(dataname,'ccaals',k,numpasses,ITERS,RESTART,rx,ry,Kcap)
% SCCA(dataname,'meg',k,numpasses,ITERS,RESTART,rx,ry,Kcap)
plotobjV(dataname,methods,k,numpasses,ITERS,'avg',0)
plotobjV(dataname,methods,k,numpasses,ITERS,'avg',1)
