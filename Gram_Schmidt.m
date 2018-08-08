%% Gram_Schmidt(A) produces an orthonormal basis for the subspace spanned by 
% the vectors A=[a1,a2,...,an] using Gram-Schmidt's orthogonalization
% procedure
%

function [U]=Gram_Schmidt(A)

[m,n]=size(A);
if(norm(A-zeros(m,n))<1e-10)
    error('Zero Vector Basis: no basis exist');
elseif(n==1)
    U=A(1:m,1)/norm(A(1:m,1));
else
    if(is_orthonormal(A)==1)
        U=A;
        return;
    end
    if(rank(A)~=n)
        A=getref(A);
    end
    [m,n]=size(A);
    U=A(1:m,1)/norm(A(1:m,1));
    for i=2:n
        u=A(1:m,i);
        v=u;
        for j=1:(i - 1)
            v=v-(u'*U(1:m,j))*U(1:m,j);
        end
        U(:,i)=v/norm(v);
    end
end
end

%% GETREF(A) returns pivot columns from reduced echelon form
%

function U=getref(A)
[m,n]=size(A);
flag = 0;
if(n==2)
    multiple=A(1,2)/A(1,1);
    count=0;
    for i=1:m
        if(A(i,2)/A(i,1) == multiple)
            count=count+1;
        end
    end
    if(count==m)
        U=A(:,1);
        flag=1;
    end
end
if(flag==0)
    [~,pivot_columns]=ref(A);
    for i=1:size(pivot_columns,2)
        U(:,i)=A(:,pivot_columns(1,i));
    end
end
end


%% REF The (R)educed (E)chelon (F)orm of A.
%       Computes the reduced row echelon form of A using partial pivoting.
%
%       Formats:   U = ref(A)
%                  U = ref(A,tol)     User specifies tolerance to determine
%                                     when an entry should be zero.
%                  [U,pivcol] = ref(A)    Also lists the pivot columns of A.
%                  [U,pivcol,nonpivcol] = ref(A)   And the nonpivot columns.

%Written by David Lay, University of Maryland, College Park, 6/20/94
%Based on the original rref(A) program written by Cleve B. Moler in 1985.
%       Version 12/15/96

function [U,pivcol,nonpivcol] = ref(A,tol)
[m,n] = size(A);
tiny = max(m,n)*eps*max(1,norm(A,'inf'))*10;    %default tolerance for zeros
if (nargin==2), tiny = tol; end                %reset tolerance, if specified 
pivcol = [];
nonpivcol = [];
U = A;
i = 1;                                  %row index
j = 1;                                  %column index
while((i <= m) && (j <= n))
   [x,k] = max(abs(U(i:m,j))); p = k+i-1; %value and row index of next pivot.
   if (x <= tiny)                       % This column is negligible.
      U(i:m,j) = zeros(m-i+1,1);        % So clean up the entries.
      nonpivcol = [nonpivcol j]; %#ok<AGROW>
      j = j + 1;                        % Pivot row p must be recalculated.
   else                     
      U([i p],j:n) = U([p i],j:n);      % Swap the ith and pth rows.
      U(i,j:n) = U(i,j:n)/U(i,j);       % Divide pivot row by the pivot.
      for k = [1:i-1 i+1:m]             % Replacement operations on other rows.
         U(k,j:n) = U(k,j:n) - U(k,j)*U(i,j:n);
      end
      pivcol = [pivcol j]; %#ok<AGROW>
      i = i + 1;
      j = j + 1;
   end
end
nonpivcol = [nonpivcol j:n];
end

%% is_orthonormal(A) returns 1 if a set of vectors is orthonormal. 
%
%

function FLAG = is_orthonormal(A)
[m,n]=size(A);
TOL = 1e-10;
FLAG=1; 
if(norm(A-zeros(m,n))<1e-10)
    error('Zero Vector Basis: no basis exist');
elseif(n==1)
    if(abs(norm(A)-1)>TOL)
        FLAG=0;
        return;
    end
else
    for i=1:n
        if(abs(norm(A(:,i))-1)>TOL)
            FLAG=0;
            return;
        end
    end
	FLAG=is_orthogonal(A);
end
end


%% is_orthogonal_set(A) returns 1 if a set of vectors is orthogonal. 
%
%
function FLAG=is_orthogonal(A)
[~,n]=size(A);
TOL=1e-10;
FLAG=1;
if(n==1)
    return;
else
    for i=1:n
        for j=1:i-1
            if(abs(A(:,i)'*A(:,j))>TOL)
                FLAG=0;
                return;
            end
        end
    end
end
end


