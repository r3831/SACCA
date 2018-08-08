%% UPDATESVD_CXY updates the SVD of the cross-covariance matrix Cxy given 
% a new sample-pair (x,y) using an incremental SVD approach
% Usage:
% Inputs:  U,S,V - current SVD, U*diag(S)*V'
%          x,y   - new sample pair
%          RANK  - maximum SVD-rank desired
% Outputs: U,S,V - updated SVD
%
%%

function [U,S,V]=updateSVD_Cxy(U,S,V,x,y,RANK) 

% m=length(x);         % Number of new columns
% n=length(y);         % Number of new columns
% if(m~=size(U,1)||(n~=size(V,1))) % Check dimensionality
%     error('Dimensions of SVD are incompatible with data-dimensionality\n');
% end

xproj=U'*x;          % Project x onto left singular subspace (rx1 vector)
xres=x-U*xproj;      % Residual x not explained by left singular subspace (mx1 vector)
xresnorm=norm(xres); % Norm of the residual
if(xresnorm>1e-6)
    xres=xres/xresnorm;  % Normalize the residual to get a unit vector normal to U
end

yproj=V'*y;          % Project y onto right singular subspace (rx1 vector)
yres=y-V*yproj;      % Residual y not explained by left singular subspace (nx1 vector)
yresnorm=norm(yres); % Norm of the residual
if(yresnorm>1e-6)
    yres=yres/yresnorm;  % Normalize the residual to get a unit vector normal to V
end

Q=[diag(S)+(xproj*yproj') yresnorm*xproj;...
    xresnorm*yproj' xresnorm*yresnorm]; % Form the (r+1)x(r+1) matrix for inner SVD
[U2,S2,V2]=svd(Q,'econ'); % Inner SVD: U2 is (r+1)x(r+1), S2 is (r+1)x(r+1), V2 is (r+1)x(r+1)
U=[U xres]*U2;       % U is mx(r+1).
S=diag(S2);                % S is (r+1)x1
V=[V yres]*V2;       % V is nx(r+1).
if(nargin>5) % TRUNCATE the SVD to the given RANK
    if(sum(S>0)>RANK)  % Check if non-zero singular values exceed RANK
        %disp('here')
        [~,idx]=sort(S,'descend'); % Sort the singular values in descending order
        idx=idx(1:RANK);  % Only top RANK singular values and vectors are needed
        U=U(:,idx);       % Top left-singular vectors
        V=V(:,idx);       % Top right-singular vectors
        S=S(idx);  % Top singular values
    end % End SVD truncation
end % End-IF
end % End-function
