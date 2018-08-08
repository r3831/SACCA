function [ U, S ] = warmuth_capping( U, S, k)

bd = 1/k ;
len = length(S);

% normalize the eigenvalues
%S = S/sum(S);
% disp(S(S>0))
% sorting eigenvalues 
[S,idx]=sort(S,'descend');
%[S,idx]=sort(S,'ascend');
%move to outside of the function
U=U(:,idx);
% S(S<0.0001) = 0;
% idx = find(S);
% S = S(idx);
% U = U(:,idx);
% S = S./sum(S);

for i=1:len-1
    if S(i) > bd
        S(i) = bd;
        S(i+1:end)=((k-i)/k)*(S(i+1:end)/sum(S(i+1:end)));
    elseif S(i) < bd
        break;
    end
end
%S(S<eps) = eps; %what is this?
