%% updateEig_Cx updates the eigendecomposition of the covariance matrix Cxy
% given a new sample x using a rank-1 update approach
% Usage:
% Inputs:  U,S - current eigendecomposition  
%          x   - new sample
%          RANK  - maximum decomposition-rank desired
% Outputs: U,S - updated eidendecomposition
%
%%

function [ U, S ] = updateEig_Cx( U, S, isamp, x, RANK )

% matrix = U * diag(S) * U'
% output = input + x * x'

d = size( U, 1 );
k = size( U, 2 );

S = S * (isamp - 1);

x_projected = U' * x;    % in the uu basis
x_leftover = x - U * x_projected;

x_leftover_norm = norm( x_leftover );

if ( ( x_leftover_norm > 1e-6 ) )

	uu_updated = [ U, x_leftover / x_leftover_norm ];
	ss_updated = [ diag( S ) + (x_projected * x_projected'), ...
        x_projected * x_leftover_norm ; ...
        x_projected' * x_leftover_norm, ...
        x_leftover_norm ^ 2 ];

	% svd deals better with repeated eigenvalues than eig
	% however, for negative eigenvalues, the left and right singular vectors may have different signs (so we multiply the singular values by left * right')
	% we may have negative eigenvalues since Warmuth's algorithm works in the log domain
	[ left, values, right ] = svd( ss_updated );
	U = uu_updated * left;
    %comment the line below out if something is wrong
	S = diag( values ) .* diag( left * right' );
    %S = diag(values);
    [~, idx] = sort(diag(S),'descend');
    idx=idx(1:min(RANK,length(idx)));
    S = S(idx);
    U = U(:,idx);
   

else

	uu_updated = U;
	ss_updated = diag( S ) + (x_projected * x_projected');

	% svd deals better with repeated eigenvalues than eig
	% however, for negative eigenvalues, the left and right singular vectors may have different signs (so we multiply the singular values by left * right')
	% we may have negative eigenvalues since Warmuth's algorithm works in the log domain
	[ left, values, right ] = svd( ss_updated );
	U = uu_updated * left;
	S = diag( values ) .* diag( left * right' );

    [~,idx]=sort(S,'descend'); % Sort the eigenvalues in descending order
    idx=idx(1:min(RANK,length(idx)));  % Only top RANK singular values and vectors are needed
    U=U(:,idx);       % Top eigenvectors
    S=S(idx);  % Top eigenvalues
    
    
end

S = S/isamp;
