function [ W, S ] = meg_pls( k, W, S, x, y, eta , RANK)

% update matrix = exp( log(matrix) + eta*[0, x*y'; y*x', 0] )
% bound = 1 / ( k );
logbound = 0;
S = log(S);

[W, S] = update_eigendecomposition(logbound, W, S, [x; y], 0.5 * eta);
[W, S] = update_eigendecomposition(logbound, W, S, [x; -y], -0.5 * eta);
S = exp(S);
S=S./sum(S);

% cap the maximum eigenvalue to bound
[ W, S ] = warmuth_capping( W, S, k );

