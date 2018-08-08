function [beta] = svrg_reg(X,Y,m,mstage,beta,lambda)
   
[n,p] = size(X);
beta_s = beta;
eta = 0.5/max(sum(X.^2,2));
rng('default');

for i=1:mstage
    v = -X'*(Y-X*beta_s)/n + lambda*beta_s;
    beta_a = zeros(p,1);   
    for j=1:m
        s = randi(n);
        jgradient = -X(s,:)'*(Y(s) - X(s,:)*beta) + lambda*beta;
        sgradient = -X(s,:)'*(Y(s) - X(s,:)*beta_s) + lambda*beta_s;
        vk = jgradient - sgradient + v;
        beta = beta - eta*vk;
        beta_a = beta_a + (1/m)*beta;
    end;
    beta_s = beta_a;
end

end