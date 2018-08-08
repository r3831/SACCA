function [u, v, corr_curve, nep] = CCA_als_svrg_reg(X,Y,lambda)
   [n,p1] = size(X);
   [~,p2] = size(Y);
   rng('default');
   Sx = X'*X/n+lambda*eye(p1);
   Sy = Y'*Y/n+lambda*eye(p2);
   Sxy = X'*Y/n;
   u = randn(p1,1);
   u = u/sqrt(u'*Sx*u);
   v = randn(p2,1);
   v = v/sqrt(v'*Sy*v);
   u_old = u;
   v_old = v;
   miter = 6;
   for i = 1:miter
       u = svrg_reg(X,Y*v,n,10,u_old,lambda);
       %u = Sx\(Sxy*v_old);
       u_old = u;
       u = u/sqrt(u'*Sx*u);
       v = svrg_reg(Y,X*u,n,10,v_old,lambda);
       %v = Sy\(Sxy'*u_old);
       v_old = v;
       v = v/sqrt(v'*Sy*v);
       corr_curve(i) = sum(diag(u'*Sxy*v));
%       fprintf('ALS_VR: %.15f\n',corr_curve(i));
   end
   nep = [1:miter].*13;
end