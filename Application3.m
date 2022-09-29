runningtime=cputime;  %record computation time

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      MODEL & DESIGN SPACE SET UP                 %        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Two variables: x1, x2
% x1 in {0, 1} 
% x2 in [-1, 1]

N1 = 201; % number of unique values for x2
N = 201*2 % total number of design points

[dp1 dp2] = ndgrid([0 1], linspace(-1, 1, N1));
dp = [dp1(:) dp2(:)];

% Saving z(x, thetastar) and z(x, thetastar)*z(x, thetastar)'
% for each x in design points for the linear model 
% y ~ 1 + x1 + x2 + x1x2 + x2^2 
all_z_vectors = [dp dp(:, 1).*dp(:, 2) dp(:, 2).^2];
all_z_outer_prods = bsxfun(@times, all_z_vectors', permute(all_z_vectors, [3 1 2]));

num_params = size(all_z_vectors, 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           COMPUTING SINGLE-OBJECTIVE             %
%               OPTIMAL DESIGNS                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol=1e-3; % We will treat weights of smaller than tol as 0

%%% Compute the A-optimal design for the linear model (Phi1) %%%
cvx_begin
    cvx_precision high
    variable A_opt_weights(N)
    expression inf_mat(num_params, num_params)
    
    for j = 1:N 
        inf_mat = inf_mat + A_opt_weights(j)*squeeze(all_z_outer_prods(:, j, :));
    end
    
    minimize ( trace_inv(inf_mat))   

    0 <= A_opt_weights <= 1;
    sum(A_opt_weights) == 1;
cvx_end


A_opt_design=[dp(find(A_opt_weights>tol), :) A_opt_weights(find(A_opt_weights>tol))]
A_opt_obj = cvx_optval;

%%% Compute the E-optimal design for the linear model (Phi2) %%%
cvx_begin
    cvx_precision high
    variable E_opt_weights(N)
    expression inf_mat(num_params, num_params)
    
    for j = 1:N 
        inf_mat = inf_mat + E_opt_weights(j)*squeeze(all_z_outer_prods(:, j, :));
    end
    
    minimize ( -lambda_min(inf_mat))   

    0 <= E_opt_weights <= 1;
    sum(E_opt_weights) == 1;
cvx_end

E_opt_design=[dp(find(E_opt_weights>tol), :) E_opt_weights(find(E_opt_weights>tol))]
E_opt_obj = cvx_optval;

%%% Compute the c-optimal design for the linear model (Phi3) %%
c=[0 0 0 1]'

cvx_begin
    cvx_precision high
    variable c_opt_weights(N)
    expression inf_mat(num_params, num_params)
    
    for j = 1:N 
        inf_mat = inf_mat + c_opt_weights(j)*squeeze(all_z_outer_prods(:, j, :));
    end
    
    minimize ( matrix_frac(c, inf_mat))   

    0 <= c_opt_weights <= 1;
    sum(c_opt_weights) == 1;
cvx_end

c_opt_design=[dp(find(c_opt_weights>tol), :) c_opt_weights(find(c_opt_weights>tol))]
c_opt_obj = cvx_optval;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      COMPUTE MAXIMIN OPTIMAL DESIGN              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cvx_begin 
    cvx_precision high 
    variable maximin_weights(N) 
    variable t
    expression inf_mat(num_params, num_params) 

    for j = 1:N 
        inf_mat = inf_mat + maximin_weights(j)*squeeze(all_z_outer_prods(:, j, :));
    end

    minimize t 
    
    trace_inv(inf_mat) - A_opt_obj*t <= 0;
    -lambda_min(inf_mat) + (-E_opt_obj)*inv_pos(t) <= 0;
    matrix_frac(c, inf_mat) - c_opt_obj*t <= 0;
    t >= 0; 
    maximin_weights >= 0; 
    sum(maximin_weights) == 1;
cvx_end

maximin_design = [dp(find(maximin_weights > tol), :) maximin_weights(find(maximin_weights > tol))]
maximin_obj = cvx_optval; 

[V D] = eig(inf_mat);
inv_information_mat = inv(inf_mat);


A_efficiency = A_opt_obj/trace(inv_information_mat)
E_efficiency = -D(1,1)/E_opt_obj
c_efficiency = c_opt_obj / (c'*inv_information_mat*c)

t % optimal value of t
1/t % this should match the smallest of the three efficiencies

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    VERIFY OPTIMALITY OF THE MAXIMIN OPTIMAL DESIGN   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set relaxation parameter for the inequality
tol_lp = 1e-4; 

d1_vect = zeros(N,1);
d2_vect = zeros(N,1);
d3_vect = zeros(N,1);

c_outer_prod = inv_information_mat*c*c'*inv_information_mat;

for j = 1:N
    d1_vect(j) = -trace(inv_information_mat) + ... 
        trace(inv_information_mat*squeeze(all_z_outer_prods(:, j, :))*inv_information_mat);
    d2_vect(j) = (all_z_vectors(j, :)*V(:, 1))^2 - D(1, 1);
    d3_vect(j) = -c'*inv_information_mat*c + all_z_vectors(j, :)*c_outer_prod*all_z_vectors(j, :)';
end

D_mat =[d1_vect d2_vect d3_vect];
complementarity_mat = diag([(trace(inv_information_mat) - t*A_opt_obj) ... 
                            (-D(1, 1) - E_opt_obj/t) ... 
                            c'*inv_information_mat*c - t*c_opt_obj]);

equality_mat = (num_params/t)*ones(1, 3);

etas = linprog(ones(3, 1), ... 
    [D_mat' complementarity_mat' complementarity_mat']', tol_lp*ones(N+3*2, 1) , ... 
    equality_mat, 1, ... 
    zeros(3, 1), [])

max(D_mat*etas) % should be around tol_lp
abs(max(D_mat*etas) - tol_lp) % should be super small

