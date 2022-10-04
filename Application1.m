runningtime=cputime;  %record computation time

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      MODEL & DESIGN SPACE SET UP                 %        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make design points for discrete design space: 
% N = 501 equally spaced points in the interval S = [0, 15] 
[S_left_endpoint, S_right_endpoint] = deal(0, 15);
N = 501; 
design_points = linspace(S_left_endpoint,S_right_endpoint,N);

% Set number of parameters and true parameter values for
% theta = (beta1, beta2, gamma1, gamma2)'
% the model y = f(x, theta) + epsilon with 
% f(x, theta) = beta1*exp(-gamma1*x) + beta2*exp(-gamma2*x)
num_params = 4;
[beta1star,beta2star,gamma1star,gamma2star] = deal(5.25,1.75,1.34,0.13);

% To help us evaluate the expected information matrix 
% at the true parameter value and at an arbitrary design later,
% we save z(x, thetastar) for each x in design_points 
all_z_vectors = zeros(N, num_params);
all_z_vectors(:, 1) =  exp(-gamma1star*design_points);
all_z_vectors(:, 2) = exp(-gamma2star*design_points); 
all_z_vectors(:, 3) = (-beta1star*design_points)' .* all_z_vectors(:, 1);
all_z_vectors(:, 4) = (-beta2star*design_points)' .* all_z_vectors(:, 2);

% We will also save z(x, thetastar)*z(x, thetastar)' for each x in f
% design_points
all_z_outer_prods = bsxfun(@times, all_z_vectors', ...
    permute(all_z_vectors, [3 1 2]));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           COMPUTING SINGLE-OBJECTIVE             %
%               OPTIMAL DESIGNS                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol=1e-3; % We will treat weights of smaller than tol as 0

%%% Compute the weighted-A-optimal design for the model (Phi1) %%%
L_Phi1_inv = diag([beta1star,beta2star,gamma1star,gamma2star]);

cvx_begin
    cvx_precision high
    variable weighted_A_opt_weights(N)
    expression information_mat(num_params, num_params)
    
    for j = 1:N 
        information_mat = information_mat + ...
            weighted_A_opt_weights(j)*squeeze(all_z_outer_prods(:, j, :));
    end
    
    minimize trace_inv(L_Phi1_inv*information_mat*L_Phi1_inv)  
    0 <= weighted_A_opt_weights <= 1;
    sum(weighted_A_opt_weights) == 1;
cvx_end

weighted_A_opt_design=[design_points(find(weighted_A_opt_weights>tol))' ...
              weighted_A_opt_weights(find(weighted_A_opt_weights>tol))]'
weighted_A_opt_obj = cvx_optval;

%%% Compute the D-optimal design for the model (Phi2) %%%
cvx_begin
    cvx_precision high
    variable D_opt_weights(N)
    expression information_mat(num_params, num_params)
    
    for j = 1:N 
        information_mat = information_mat + ...
            D_opt_weights(j)*squeeze(all_z_outer_prods(:, j, :));
    end
    
    minimize ( -log_det(information_mat))   

    0 <= D_opt_weights <= 1;
    sum(D_opt_weights) == 1;
cvx_end

D_opt_design=[design_points(find(D_opt_weights>tol))' ...
              D_opt_weights(find(D_opt_weights>tol))]'
D_opt_obj = cvx_optval;


%%% Compute the I-optimal design for the model (Phi3) %%%

% Make the L-matrix for coding integrated optimality criterion (Phi3): 
% First we use Matlab's Symbolic Math Toolbox to evaluate the definite 
% integral in the expression for L^2 
syms beta1 beta2 gamma1 gamma2 x

f = [exp(-gamma1*x); exp(-gamma2*x); ...
     -beta1*x*exp(-gamma1*x); -beta2*x*exp(-gamma2*x)];

F = f*f.';

L2_expression = int(F,x,2,10);

% This makes a .m file containing a function in your current folder
L2_function = matlabFunction(L2_expression,'File','L2_function', ... 
                    'Vars',[beta1 beta2 gamma1 gamma2]); 

L_Phi2_squared = L2_function(beta1star,beta2star, gamma1star,gamma2star);  

% Compute the inverse of L-matrix via eigendecomposition 
[V,D]=eig(L_Phi2_squared);
L_Phi2 = V*sqrt(D)*V';
L_Phi2_inv =V*sqrt(inv(D))*V';

cvx_begin
    cvx_precision high
    variable I_opt_weights(N)
    expression information_mat(num_params, num_params)
    
    for j = 1:N 
        information_mat = information_mat + ...
            I_opt_weights(j)*squeeze(all_z_outer_prods(:, j, :));
    end

    minimize trace_inv(L_Phi2_inv*information_mat*L_Phi2_inv)   
    0 <= I_opt_weights <= 1;
    sum(I_opt_weights) == 1;
cvx_end

I_opt_design=[design_points(find(I_opt_weights>tol))' ...
              I_opt_weights(find(I_opt_weights>tol))]'
I_opt_obj = cvx_optval;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           EFFICIENCY CONSTRAINTS SET-UP          %        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[min_D_eff, min_I_eff] = deal(0.9, 0.8);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           COMPUTE EFFICIENCY CONSTRAINED         %
%                  OPTIMAL DESIGN                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cvx_begin 
    cvx_precision high
    variable constrained_opt_weights(N) 
    expression information_mat(num_params, num_params)
    
    for j = 1:N 
        information_mat = information_mat + ...
            constrained_opt_weights(j)*squeeze(all_z_outer_prods(:, j, :));
    end

    minimize trace_inv(L_Phi1_inv*information_mat*L_Phi1_inv) 
    0 <= constrained_opt_weights <= 1;
    sum(constrained_opt_weights) == 1;
    -log_det(information_mat) <= D_opt_obj - num_params*log(min_D_eff);
    trace_inv(L_Phi2_inv*information_mat*L_Phi2_inv) <= I_opt_obj/min_I_eff;
cvx_end

constrained_opt_design = ... 
    [design_points(find(constrained_opt_weights>tol))' ...
              constrained_opt_weights(find(constrained_opt_weights>tol))]'

% Cheng and Yang (2019) report Phi_1-efficiency = 0.8692, 
% Phi_2-efficiency = 0.9000, and Phi_3-efficiency = 0.8001 
% for their efficiency-constrained optimal design in Table 3 

% Phi_1-efficiency of the efficiency-constrained design we found: 
Phi1_efficiency = weighted_A_opt_obj/ ...
    trace(inv(L_Phi1_inv*information_mat*L_Phi1_inv))
% Phi_2-efficiency of the efficiency-constrained design we found: 
Phi2_efficiency = (exp(D_opt_obj)^(1/num_params))/...
    (exp(-log(det(information_mat)))^(1/num_params))
% Phi_3-efficiency of the efficiency-constrained design we found: 
Phi3_efficiency = I_opt_obj/trace(inv((L_Phi2_inv*information_mat*L_Phi2_inv)))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      VERIFY OPTIMALITY OF THE EFFICIENCY         %
%          CONSTRAINED OPTIMAL DESIGN              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set relaxation parameter for the inequality
tol_lp = 1e-4; 

% Find Lagrange multipliers using linear programming
d1_vect=zeros(N,1);
d2_vect=zeros(N,1);
d3_vect=zeros(N,1);
     
L_Phi1 = inv(L_Phi1_inv);
inv_information_mat = inv(information_mat);

for j = 1:N
    jth_z_outer_prod = squeeze(all_z_outer_prods(:, j, :));

    d1_vect(j) = -trace(inv_information_mat*L_Phi1*L_Phi1') + ... 
    trace(L_Phi1*inv_information_mat*jth_z_outer_prod*inv_information_mat*L_Phi1);
    d2_vect(j) = trace(inv_information_mat*jth_z_outer_prod) - num_params; 
    d3_vect(j) = -trace(inv_information_mat*L_Phi2*L_Phi2') + ... 
    trace(L_Phi2*inv_information_mat*jth_z_outer_prod*inv_information_mat*L_Phi2);
end

B_ineq=[d2_vect d3_vect];
B_eq = zeros(2); 
B_eq(1, 1) = -log(det(information_mat)) - ... 
    (D_opt_obj - num_params*log(min_D_eff));
B_eq(2, 2) = trace(inv((L_Phi2_inv*information_mat*L_Phi2_inv))) -  ... 
    I_opt_obj/min_I_eff;

b = -d1_vect+tol_lp;
     
eta = linprog(ones(2, 1), ... 
    [B_ineq' B_eq -B_eq]', [b' tol_lp*ones(1, 4)]', ...
    [], [], zeros(2, 1), [])

d_constr_vect = d1_vect + eta(1)*d2_vect + eta(2)*d3_vect;

max(d_constr_vect) % should be around tol_lp

abs(max(d_constr_vect) - tol_lp) % Should be very very small 

resulttime=cputime-runningtime

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   MAKE PLOTS                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FIGURE 1
tiledlayout(1,4,'TileSpacing','compact', 'Padding','compact')
zero_line = zeros(N, 1);

ax1=nexttile;
plot(design_points,d1_vect,design_points,zero_line,'r--','LineWidth',1)
ax1.FontSize = 8;
ax1.LineWidth = 1;
title(ax1,'(a) $\phi_1$-optimality', 'Interpreter', 'latex', ...
        'FontSize', 9)
xlabel('$u_i$', 'Interpreter', 'latex', ...
        'FontSize', 10)
ylabel('$d_{\phi_1, f}(u_i, {\bf w}^{*m}$)', 'Interpreter', 'latex', ...
        'FontSize', 9)

ax2=nexttile;
plot(design_points,d2_vect,design_points,zero_line,'r--','LineWidth',1)
ax2.FontSize = 8;
ax2.LineWidth = 1;
title(ax2,'(b) $\phi_2$-optimality', 'Interpreter', 'latex', ...
        'FontSize', 9)
xlabel('$u_i$', 'Interpreter', 'latex', ...
        'FontSize', 10)
ylabel('$d_{\phi_2, f}(u_i, {\bf w}^{*m}$)', 'Interpreter', 'latex', ...
        'FontSize', 9)
    
ax3=nexttile;
plot(design_points,d3_vect,design_points,zero_line,'r--','LineWidth',1)
ax3.FontSize = 8;
ax3.LineWidth = 1;
title(ax3,'(c) $\phi_3$-optimality', 'Interpreter', 'latex', ...
        'FontSize', 9)
xlabel('$u_i$', 'Interpreter', 'latex', ...
        'FontSize', 10)
ylabel('$d_{\phi_3, f}(u_i, {\bf w}^{*m}$)', 'Interpreter', 'latex', ...
        'FontSize', 9)
    
ax4=nexttile;
plot(design_points,d_constr_vect,design_points,zero_line,'r--','LineWidth',1)
ax4.FontSize = 8;
ax4.LineWidth = 1;
title(ax4,{['(d) Multi-objective'], ['optimality']}, 'Interpreter', 'latex', ...
        'FontSize', 9)
xlabel('$u_i$', 'Interpreter', 'latex', ...
        'FontSize', 10)
ylabel('$d_1(u_i, {\bf w}^{*m}) + \sum \limits_{k=2}^3 \eta_k^* d_k(u_i, {\bf w}^{*m})$', ...
        'Interpreter','latex','FontSize',9)
limsy=get(gca,'YLim');
ylim([limsy(1) 10])

h = gcf;
set(h, 'PaperUnits','centimeters');
set(h, 'Units','centimeters');

pos=get(h,'Position');
set(h, 'Position', [pos(1),pos(2), 17, 5]);

pos=get(h,'Position');
set(h, 'PaperSize', [pos(3) pos(4)]);
set(h, 'PaperPositionMode', 'manual');
set(h, 'PaperPosition',[0 0 pos(3) pos(4)]);

print('Figure1', '-dpdf')
