
%% Extracts the best maximal kk-dimensional subspace from a solution
%
% kk - the dimension of the subspace which we seek
% left, right, values - "nontrivial" signular vectors and singular values
%                   of the iterate
%%

function [ left_solution, right_solution ] = msg_pls_solution( kk, left, values, right )
[ ~, indices ] = sort( values, 'descend' );
left_solution = left( :, indices( 1:kk ) );
right_solution = right( :, indices( 1:kk ) );

% left_solution  = left_solution  / sqrtm( left_solution'  * left_solution  );
% right_solution = right_solution / sqrtm( right_solution' * right_solution );

end
