function [B,index,angles] = sort_library_by_angle(A)
%        [B,index,angles] = sort_library_by_angle(A)
%
% B = A(index,:) where index in the column index of A ordered by inreasing
% minimum angle with every other colum
%
%
% % Author: Jose Bioucas Dias. June, 2011

BIG = 1e10;

[L,m] = size(A);  % L = number of bands; m = number of materilas

%normalize A
nA = sqrt(sum(A.^2));
A_norm = A./repmat(nA,L,1);

% compute angles
angles = abs(acos(A_norm'*A_norm))*180/pi;
angles = angles + BIG*diag(ones(1,m));

% compute min angles between a given column and every other column
[min_angles index_rows] = min(angles);
% sort columns by increasing angles
[angles, index] = sort(min_angles);

B = A(:,index);





