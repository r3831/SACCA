function [ U, S ] = update_eigendecomposition( shift, U, S, x, eta )

% matrix = U * diag(S) * U'
% output = matrix + eta*x*x'
d = size( U, 1 );
k = size( U, 2 );
%disp(k)

x_proj = U' * x;    % in the U basis
x_res = x - U * x_proj;
x_res_norm = norm( x_res );

%disp(x_res_norm)
%disp(k)
if ( ( k < d ) && ( x_res_norm > 1e-6 ) )

	uu_updated = [ U, x_res / x_res_norm ];
	ss_updated = [ diag( S ) + eta * (x_proj * x_proj'), ...
        eta * x_proj * x_res_norm ; eta * x_proj' * x_res_norm, shift+eta * x_res_norm ^ 2 ];

    %disp(k)
    %disp(x_res_norm)
	% svd deals better with repeated eigenvalues than eig
	% however, for negative eigenvalues, the left and right singular vectors may have different signs (so we multiply the singular values by left * right')
	% we may have negative eigenvalues since Warmuth's algorithm works in the log domain
    %disp(ss_updated)
	[ left, values, right ] = svd( ss_updated );
	U = uu_updated * left;
	S = diag( values ) .* diag( left * right' );

else

	uu_updated = U;
	ss_updated = diag( S ) + eta * (x_proj * x_proj');

	% svd deals better with repeated eigenvalues than eig
	% however, for negative eigenvalues, the left and right singular vectors may have different signs (so we multiply the singular values by left * right')
	% we may have negative eigenvalues since Warmuth's algorithm works in the log domain
	[ left, values, right ] = svd( ss_updated );
	U = uu_updated * left;
	S = diag( values ) .* diag( left * right' );

end
