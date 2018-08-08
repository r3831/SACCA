function [ U, V ] = warmuth_pls_solution( k, U, S, V )

[ ~, idx ] = sort( S, 'descend' );
%[ ~, idx ] = sort( S, 'ascend' );

U  = U(  :, idx( 1:k ) );
V = V( :, idx( 1:k ) );

U  = U  / sqrtm( U'  * U  );
V = V / sqrtm( V' * V );
