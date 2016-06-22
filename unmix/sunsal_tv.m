function [U,res,rmse] = sunsal_tv(M,Y,varargin)

%% [U,res,rmse] = sunsal_tv(M,y,varargin)
%
%  SUNSAL_TV ->  sparse unmixing with TV via variable splitting and augmented
%  Lagragian methods  intoduced in
%
%  M.-D. Iordache, J. Bioucas-Dias, and A. Plaza, "Total variation spatial
%  regularization for sparse hyperspectral unmixing", IEEE Transactions on
%  Geoscience and Remote Sensing, vol. PP, no. 99, pp. 1-19, 2012.
%
%
%% --------------- Description ---------------------------------------------
%
%  SUNSAL_TV solves the following l_2 + l_{1,1} + TV optimization problem:
%
%     Definitions
%
%      A  -> L * n; Mixing matrix (Library)
%      X  -> n * N; collection of N fractional vectors; each column  of X
%                   contains the fraction of a correspondent  pixel
%
%      Optimization problem
%
%    min  (1/2) ||A X-Y||^2_F  + lambda_1  ||X||_{1,1}
%     X                         + lambda_tv ||LX||_{1,1};
%
%
%    where
%
%        (1/2) ||A X-Y||^2_F is a quadratic data misfit term
%
%        ||X||_{1,1} = sum_i ||X(:,i)||_1, for i = 1,...,N.
%                      is the standard l1 regularizer
%
%        ||LX||_{1,1} is the TV (non-isotropic or isotropic regularizer)
%
%
%         L is a linear operator that computes the horizontal and the
%         vertical differences on each  band of X.  Let Lh: R^{n*N}-> R^{n*N}
%         be a linear operator that computes the horizontal first order
%         differences per band. LhX  computes a matrix of the same size of X
%         (we are assuming cyclic boundary), where [LhX](i,j) = X(i,h(j))-X(i,j),
%         where h(j) is the index of pixel on the right hand side of j.
%
%         For the vertical differnces, we have a similar action of Lv:
%         [LvX](i,j) = X(v(i),j)-X(i,j), where  v(i) is the index of pixel
%         on the top hand side of j.
%
%         We consider tow types of Total variation:
%
%         a)  Non-isotropic:  ||LX||_{1,1} := ||[Lh; Lv]X||_{1,1}
%
%         b) Isotropic:  ||LX||_{1,1}  := ||(LhX, LvX)||_11,
%             where   |||(A,B)||_{1,1} := |||sqrt(A.^2 + B.^2)||_{1,1}
%
%
% -------------------------------------------------------------------------
%
%
%
%    CONSTRAINTS ACCEPTED:
%
%    1) Positivity X(:,i) >= 0, for i=1,...,N
%    2) Sum-To-One sum( X(:,i)) = 1, for for i=1,...,N
%
%
%    NOTES:
%
%       1) If X is a matrix and lambda_TV = 0, SUNSAL_TV solves
%           columnwise independent optimizations.
%
%       2) If both the Positivity and Sum-To-One constraints are active,
%          then we have ||X||_{1,1} = n and therefore this regularizer
%          is useless.
%
%
%% -------------------- Line of Attack  -----------------------------------
%
%  SUNSAL_TV solves the above optimization problem by introducing a variable
%  splitting and then solving the resulting constrained optimization with
%  the augmented Lagrangian method.
%
%
%   The initial problem is converted into
%
%    min  (1/2) ||A X-Y||^2_F  + i_R_+(X)
%     X                        + i_S(X)
%                              + lambda_1  ||X||_{1,1}
%                              + lambda_tv ||LX||_{1,1};
%
%
%   where i_R_+ and i_S are the indicator functions of the set R_+ and
%   the probability simplex, respecively, applied to the columns ox X.
%
%
%  Then, we apply the following variable splitting
%
%
%    min  (1/2) ||V1-Y||^2     + i_R_+(V2)
%  U,V1, .... V7               + i_S(V3)
%                              + lambda_1  ||V4||_{1,1}
%                              + lambda_tv ||V6||_{1,1};
%
%     subject to:  AU   = V1
%                  U    = V2
%                  U    = V3
%                  U    = V4
%                  U    = V5
%                  LV5  = V6
%
%
%  For details see
%
%
%  J. Bioucas-Dias and M. Figueiredo, “Alternating direction algorithms for
%  constrained sparse regression: Application to hyperspectral unmixing”,
%  in  2nd  IEEE GRSS Workshop on Hyperspectral Image and Signal Processing-WHISPERS'2010,
%  Raykjavik, Iceland, 2010.
%
%
%  M.-D. Iordache, J. Bioucas-Dias, and A. Plaza, "Total variation spatial
%  regularization for sparse hyperspectral unmixing", IEEE Transactions on
%  Geoscience and Remote Sensing, vol. PP, no. 99, pp. 1-19, 2012.
%
%  M.-D. Iordache, J. Bioucas-Dias and A. Plaza, "Sparse unmixing
%  of hyperspectral data", IEEE Transactions on Geoscience and Remote Sensing,
%  vol. 49, no. 6, pp. 2014-2039, 2011.
%
%  M. V. Afonso, J. Bioucas-Dias, and M. Figueiredo, “An Augmented
%  Lagrangian Approach to the Constrained Optimization Formulation of
%  Imaging Inverse Problems”, IEEE Transactions on Image Processing,
%  vol. 20, no. 3, pp. 681-695, 2011.
%
%
%
%
% ------------------------------------------------------------------------
%%  ===== Required inputs =============
%
%  M - [L(observations) * n (variables)] system matrix (usually a library)
%
%  Y - matrix with  L(observation) x N(pixels).
%
%
%%  ====================== Optional inputs =============================
%
%
%  'LAMBDA_1' - regularization parameter for l11 norm.
%               Default: 0;
%
%  'LAMBDA_TV' - regularization parameter for TV norm.
%                Default: 0;
%
%  'TV_TYPE'   - {'iso','niso'} type of total variation:  'iso' ==
%                isotropic; 'n-iso' == non-isotropic; Default: 'niso'
%
%  'IM_SIZE'   - [nlins, ncols]   number of lines and rows of the
%                spectral cube. These parameters are mandatory when
%                'LAMBDA_TV' is  passed.
%                Note:  n_lin*n_col = N
%
%
%  'AL_ITERS' - (double):   Minimum number of augmented Lagrangian iterations
%                           Default 100;
%
%
%  'MU' - (double):   augmented Lagrangian weight
%                           Default 0.001;
%
%
%
%  'POSITIVITY'  = {'yes', 'no'}; Default 'no'
%                  Enforces the positivity constraint: x >= 0
%
%  'ADDONE'  = {'yes', 'no'}; Default 'no'
%               Enforces the positivity constraint: x >= 0
%
%  'TRUE_X'  - [n (variables), N (pixels)] original data in matrix format.
%              If  the XT (the TRUE X) is inputted, then the RMSE is
%              ||X-XT||computed along the iterations
%
%
%  'VERBOSE'   = {'yes', 'no'}; Default 'no'
%
%                 'no' - work silently
%                 'yes' - display warnings
%
%%  =========================== Outputs ==================================
%
% U  =  [nxN] estimated  X matrix
%
%

%%
% ------------------------------------------------------------------
% Author: Jose Bioucas-Dias, January, 2010.
%
%
% Modifications:
%
%   Jose Bioucas-Dias, July 2010:  -> Introduction of isotropic TV.
%
%
%% -------------------------------------------------------------------------
%
% Copyright (January, 2011):        José Bioucas-Dias (bioucas@lx.it.pt)
%
% SUNSAL_TV is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------

%%
%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 2
    error('Wrong number of required parameters');
end
% mixing matrix size
[LM,n] = size(M);
% data set size
[L,N] = size(Y);
if (LM ~= L)
    error('mixing matrix M and data set y are inconsistent');
end




%%
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
%


% 'LAMBDA_1'
%  l1 regularization
reg_l1 = 0; % absent

% 'LAMBDA_TV'
%  TV regularization
reg_TV = 0; % absent
im_size = []; % image size
tv_type = 'niso'; % non-isotropic TV

% 'AL:ITERS'
% maximum number of AL iteration
AL_iters = 1000;

% 'MU'
% AL weight
mu = 0.001;

% 'VERBOSE'
% display only sunsal warnings
verbose = 'off';

% 'POSITIVITY'
% Positivity constraint
positivity = 'no';
reg_pos = 0; % absent

% 'ADDONE'
%  Sum-to-one constraint
addone = 'no';
reg_add = 0; % absent

%

% initialization
U0 = 0;

% true X
true_x = 0;
rmse = 0;

%%
%--------------------------------------------------------------
% Local variables
%--------------------------------------------------------------


%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'LAMBDA_1'
                lambda_l1 = varargin{i+1};
                if lambda_l1 < 0
                    error('lambda must be positive');
                elseif lambda_l1 > 0
                    reg_l1 = 1;
                end
            case 'LAMBDA_TV'
                lambda_TV = varargin{i+1};
                if lambda_TV < 0
                    error('lambda must be non-negative');
                elseif lambda_TV > 0
                    reg_TV = 1;
                end
            case 'TV_TYPE'
                tv_type = varargin{i+1};
                if ~(strcmp(tv_type,'iso') | strcmp(tv_type,'niso'))
                    error('wrong TV_TYPE');
                end
            case 'IM_SIZE'
                im_size = varargin{i+1};
            case 'AL_ITERS'
                AL_iters = round(varargin{i+1});
                if (AL_iters <= 0 )
                    error('AL_iters must a positive integer');
                end
            case 'POSITIVITY'
                positivity = varargin{i+1};
                if strcmp(positivity,'yes')
                    reg_pos = 1;
                end
            case 'ADDONE'
                addone = varargin{i+1};
                if strcmp(addone,'yes')
                    reg_add = 1;
                end
            case 'MU'
                mu = varargin{i+1};
                if mu <= 0
                    error('mu must be positive');
                end
            case 'VERBOSE'
                verbose = varargin{i+1};
            case 'X0'
                U0 = varargin{i+1};
            case 'TRUE_X'
                XT = varargin{i+1};
                true_x = 1;
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end

% test for true data size correctness
if true_x
    [nr nc] = size(XT);
    if (nr ~= n) | (nc ~= N)
        error('wrong image size')
    end
end


% test for image size correctness
if reg_TV > 0
    if N ~= prod(im_size)
        error('wrong image size')
    end
    n_lin = im_size(1);
    n_col = im_size(2);
    
    % build handlers and necessary stuff
    % horizontal difference operators
    FDh = zeros(im_size);
    FDh(1,1) = -1;
    FDh(1,end) = 1;
    FDh = fft2(FDh);
    FDhH = conj(FDh);
    
    % vertical difference operator
    FDv = zeros(im_size);
    FDv(1,1) = -1;
    FDv(end,1) = 1;
    FDv = fft2(FDv);
    FDvH = conj(FDv);
    
    IL = 1./( FDhH.* FDh + FDvH.* FDv + 1);
    
    Dh = @(x) real(ifft2(fft2(x).*FDh));
    DhH = @(x) real(ifft2(fft2(x).*FDhH));
    
    Dv = @(x) real(ifft2(fft2(x).*FDv));
    DvH = @(x) real(ifft2(fft2(x).*FDvH));
    
end




%%
%---------------------------------------------
% just least squares
%---------------------------------------------
if ~reg_TV && ~reg_l1 && ~reg_pos && ~reg_add
    U = pinv(M)*Y;
    res = norm(M*X-Y,'fro');
    return
end
%---------------------------------------------
% just ADDONE constrained (sum(x) = 1)
%---------------------------------------------
SMALL = 1e-12;
B = ones(1,n);
a = ones(1,N);

if  ~reg_TV && ~reg_l1 && ~reg_pos && reg_add
    F = M'*M;
    % test if F is invertible
    if rcond(F) > SMALL
        % compute the solution explicitly
        IF = inv(F);
        U = IF*M'*Y-IF*B'*inv(B*IF*B')*(B*IF*M'*Y-a);
        res = norm(M*U-Y,'fro');
        return
    end
    % if M'*M is singular, let sunsal_tv run
end


%%
%---------------------------------------------
%  Constants and initializations
%---------------------------------------------

% number of regularizers
n_reg =  reg_l1 + reg_pos + reg_add + reg_TV;

IF = inv(M'*M + n_reg*eye(n));

%%
%---------------------------------------------
%  Initializations
%---------------------------------------------

% no intial solution supplied
if U0 == 0
    U = IF*M'*Y;
end

% what regularizers ?
%  1 - data term
%  2 - positivity
%  3 - addone
%  4 - l1
%  5 - TV


index = 1

% initialize V variables
V = cell(1 + n_reg,1);

% initialize D variables (scaled Lagrange Multipliers)
D = cell(1 + n_reg,1);


%  data term (always present)
reg(1) = 1;             % regularizers
V{index} = M*U;         % V1
D{1} = zeros(size(Y));  % Lagrange multipliers

% next V
index = index + 1;
% POSITIVITY
if reg_pos == 1
    reg(index) = 2;
    V{index} = U;
    D{index} = zeros(size(U));
    index = index +1;
end
% ADDONE
if reg_add == 1
    reg(index) = 3;
    V{index} = U;
    D{index} = zeros(size(U));
    index = index +1;
end
%l_{1,1}
if reg_l1 == 1
    reg(index) = 4;
    V{index} = U;
    D{index} = zeros(size(U));
    index = index +1;
end
%TV
% NOTE: V5, V6, D5, and D6 are represented as image planes
if reg_TV == 1
    reg(index) = 5;
    % V5
    V{index} = U;
    D{index} = zeros(size(U));
    
    % convert X into a cube
    U_im = reshape(U',im_size(1), im_size(2),n);
    
    % V6 create two images per band (horizontal and vertical differences)
    V{index+1} = cell(n,2);
    D{index+1} = cell(n,2);
    for i=1:n
        % build V6 image planes
        V{index+1}{i}{1} = Dh(U_im(:,:,i));   % horizontal differences
        V{index+1}{i}{2} = Dv(U_im(:,:,i));   % horizontal differences
        % build d7 image planes
        D{index+1}{i}{1} = zeros(im_size);   % horizontal differences
        D{index+1}{i}{2} = zeros(im_size);   % horizontal differences
    end
    clear U_im;
end






%%
%---------------------------------------------
%  AL iterations - main body
%---------------------------------------------
tol1 = sqrt(N)*1e-5;
i=1;
res = inf;
while (i <= AL_iters) && (sum(abs(res)) > tol1)
    
    
    % solve the quadratic step (all terms depending on U)
    Xi = M'*(V{1}+D{1});
    for j = 2:(n_reg+1)
        Xi = Xi+ V{j} + D{j};
    end
    U = IF*Xi;
    
    
    % Compute the Mourau proximity operators
    for j=1:(n_reg+1)
        %  data term (V1)
        if  reg(j) == 1
            V{j} = (1/(1+mu)*(Y+mu*(M*U-D{j})));
        end
        %  positivity   (V2)
        if  reg(j) == 2
            V{j} = max(U-D{j},0);
        end
        % addone  (project on the affine space sum(x) = 1)  (V3)
        if  reg(j) == 3
            nu_aux = U - D{j};
            V{j} = nu_aux + repmat((1-sum(nu_aux))/n,n,1);
        end
        % l1 norm  (V4)
        if  reg(j) == 4
            V{j} = soft(U-D{j},lambda_l1/mu);
        end
        % TV  (V5 and V6)
        if  reg(j) == 5
            % update V5: solves the problem:
            %    min 0.5*||L*V5-(V6+D7)||^2+0.5*||V5-(U-d5)||^2
            %      V5
            %
            % update V6: min 0.5*||V6-(L*V5-D6)||^2 + lambda_tv * |||V6||_{1,1}
            
            nu_aux = U - D{j};
            % convert nu_aux into image planes
            % convert X into a cube
            nu_aux5_im = reshape(nu_aux',im_size(1), im_size(2),n);
            % compute V5 in the form of image planes
            for k =1:n
                % V5
                V5_im(:,:,k) = real(ifft2(IL.*fft2(DhH(V{j+1}{k}{1}+D{j+1}{k}{1}) ...
                    +  DvH(V{j+1}{k}{2}+D{j+1}{k}{2}) +  nu_aux5_im(:,:,k))));
                % V6
                aux_h = Dh(V5_im(:,:,k));
                aux_v = Dv(V5_im(:,:,k));
                if strcmp(tv_type, 'niso')  % non-isotropic TV
                    V{j+1}{k}{1} = soft(aux_h - D{j+1}{k}{1}, lambda_TV/mu);   %horizontal
                    V{j+1}{k}{2} = soft(aux_v - D{j+1}{k}{2}, lambda_TV/mu);   %vertical
                else    % isotropic TV
                    % Vectorial soft threshold
                    aux = max(sqrt((aux_h - D{j+1}{k}{1}).^2 + (aux_v - D{j+1}{k}{2}).^2)-lambda_TV/mu,0);
                    V{j+1}{k}{1} = aux./(aux+lambda_TV/mu).*(aux_h - D{j+1}{k}{1});
                    V{j+1}{k}{2} = aux./(aux+lambda_TV/mu).*(aux_v - D{j+1}{k}{2});
                end
                % update D6
                D{j+1}{k}{1} =  D{j+1}{k}{1} - (aux_h - V{j+1}{k}{1});
                D{j+1}{k}{2} =  D{j+1}{k}{2} - (aux_v - V{j+1}{k}{2});
            end
            % convert V6 to matrix format
            V{j} = reshape(V5_im, prod(im_size),n)';
            
        end
        
    end
    
    
    
    % update Lagrange multipliers
    
    for j=1:(n_reg+1)
        if  reg(j) == 1
            D{j} = D{j} - (M*U-V{j});
        else
            D{j} = D{j} - (U-V{j});
        end
    end
    
    
    
    
    % compute residuals
    if mod(i,10) == 1
        st = [];
        for j=1:(n_reg+1)
            if  reg(j) == 1
                res(j) = norm(M*U-V{j},'fro');
                st = strcat(st,sprintf(' res(%i) = %2.6f',reg(j),res(j) ));
            else
                res(j) = norm(U-V{j},'fro');
                st = strcat(st,sprintf('  res(%i) = %2.6f',reg(j),res(j) ));
            end
        end
        if  strcmp(verbose,'yes')
            fprintf(strcat(sprintf('iter = %i -',i),st,'\n'));
        end
    end
    
    
    
    % compute RMSE
    if true_x
        rmse(i)= norm(U-XT,'fro');
        if  strcmp(verbose,'yes')
            fprintf(strcat(sprintf('iter = %i - ||Xhat - X|| = %2.3f',i, rmse(i)),'\n'));
        end
        
    end
    
    i=i+1;
    
end








% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
