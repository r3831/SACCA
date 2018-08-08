function S = the_projection( S, k )
%% the_projection projects with respect to trace <= k                     
% assumes S is sorted in ascending order

%% check if capping without shift is OK
SS=S;
SS(S>1)=1; SS(S<0)=0;
if sum(SS)<=k
    S=SS;
    return;
end

%% trace was bigger than k after capping ==> shf <= 0
% rule: SS(i:j) should not be capped and everything outside that range
% should be capped accordingle, i.e. SS(1:i-1)=0 and SS(j+1:l)=1
l=length(S);
for i=1:l
    for j=i:l
        SS=S;
        if i>1
            SS(1:i-1)=0;
        end
        if j<l
            SS(j+1:l)=1;
        end
        shf=(k-sum(SS))/(j-i+1);
        SS(i:j)=SS(i:j)+shf;
        %% check for consistency
        if (SS(i)>=0 ...
                && (i==1 || S(i-1)+shf<=0) ... % the order of operands is important
                && SS(j)<=1 ...
                && (j==l || S(j+1)+shf>=1) ) % the order of operands is important
            S=SS;
            return;
        end
    end
end
end
