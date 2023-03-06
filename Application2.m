runningtime=cputime;  %record computation time

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      MODEL & DESIGN SPACE SET UP                 %        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make design points for discrete design space: 
% N = 501 equally spaced points in the interval S = [0, 500] 
[S_left_endpoint, S_right_endpoint] = deal(0, 500);
N = 501; 
design_points = linspace(S_left_endpoint,S_right_endpoint,N);

% Set true parameter values for the non-linear models 
EmaxI_theta_star = [60  294  25]; 
EmaxII_theta_star = [60 340 107.14]; 
logistic_theta_star =  [49.62  290.51 150 45.51]; 

% Saving z(x, thetastar) and z(x, thetastar)*z(x, thetastar)'
% for each x in design points for the linear model y ~ 1 + x
all_z_vectors_linear = [ones(N, 1) design_points'];
all_z_outer_prods_linear = bsxfun(@times, all_z_vectors_linear', ...
    permute(all_z_vectors_linear, [3 1 2]));

% Saving z(x, thetastar) and z(x, thetastar)*z(x, thetastar)'
% for each x in design points for the EmaxI model
all_z_vectors_EmaxI = ones(N, 3);
all_z_vectors_EmaxI(:, 2) = design_points ./ (EmaxI_theta_star(3) + design_points); 
all_z_vectors_EmaxI(:, 3) = -(EmaxII_theta_star(2)*design_points) ./ ... 
    ((EmaxI_theta_star(3) + design_points).^2);
all_z_outer_prods_EmaxI = bsxfun(@times, all_z_vectors_EmaxI', ...
    permute(all_z_vectors_EmaxI, [3 1 2]));

% Saving z(x, thetastar) and z(x, thetastar)*z(x, thetastar)'
% for each x in design points for the EmaxII model
all_z_vectors_EmaxII = ones(N, 3);
all_z_vectors_EmaxII(:, 2) = design_points ./ (EmaxII_theta_star(3) + design_points); 
all_z_vectors_EmaxII(:, 3) = -(EmaxII_theta_star(3)*design_points) ./ ... 
    ((EmaxII_theta_star(3) + design_points).^2);
all_z_outer_prods_EmaxII = bsxfun(@times, all_z_vectors_EmaxII', ...
    permute(all_z_vectors_EmaxII, [3 1 2]));

% Saving z(x, thetastar) and z(x, thetastar)*z(x, thetastar)'
% for each x in design points for the logistic model
all_z_vectors_logistic = ones(N, 4);

logistic_comp = exp((logistic_theta_star(3) - design_points) / logistic_theta_star(4));

all_z_vectors_logistic(:, 2) = 1 ./ (1 + logistic_comp); 
all_z_vectors_logistic(:, 3) = (-logistic_theta_star(2)*logistic_comp) ./ ... 
    logistic_theta_star(3) ./ ((1 + logistic_comp).^2); 
all_z_vectors_logistic(:, 4) = logistic_theta_star(2)*logistic_comp .* ... 
    (logistic_theta_star(3) - design_points) ./ logistic_theta_star(4)^2 ./ ... 
    ((1 + logistic_comp).^2);

all_z_outer_prods_logistic = bsxfun(@times, all_z_vectors_logistic', ...
    permute(all_z_vectors_logistic, [3 1 2]));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           COMPUTING SINGLE-OBJECTIVE             %
%               OPTIMAL DESIGNS                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol=1e-3; % We will treat weights of smaller than tol as 0

%%% Compute the D-optimal design for the linear model (Phi1) %%%
cvx_begin
    cvx_precision high
    variable D_lin_weights(N)
    expression lin_inf_mat(2, 2)
    
    for j = 1:N 
        lin_inf_mat = lin_inf_mat + ...
            D_lin_weights(j)*squeeze(all_z_outer_prods_linear(:, j, :));
    end
    

    % Here, we reformulate problem as minimizing 
    % -det_rootn(information_mat) to avoid successive approximation; 
    % see CVX User's Guide for details    
    minimize ( -det_rootn(lin_inf_mat))   

    0 <= D_lin_weights <= 1;
    sum(D_lin_weights) == 1;
cvx_end

D_lin_design=[design_points(find(D_lin_weights>tol))' ...
              D_lin_weights(find(D_lin_weights>tol))]'
D_lin_obj = -log(det(lin_inf_mat));

%%% Compute the D-optimal design for the EmaxI model (Phi2) %%%
cvx_begin
    cvx_precision high
    variable D_EmaxI_weights(N)
    expression EmaxI_inf_mat(3, 3)
    
    for j = 1:N 
        EmaxI_inf_mat = EmaxI_inf_mat + ...
            D_EmaxI_weights(j)*squeeze(all_z_outer_prods_EmaxI(:, j, :));
    end
    
    % See earier note about log_det vs. det_rootn
    minimize ( -det_rootn(EmaxI_inf_mat))   

    0 <= D_EmaxI_weights <= 1;
    sum(D_EmaxI_weights) == 1;
cvx_end

D_EmaxI_design=[design_points(find(D_EmaxI_weights>tol))' ...
              D_EmaxI_weights(find(D_EmaxI_weights>tol))]'
D_EmaxI_obj = -log(det(EmaxI_inf_mat));

%%% Compute the D-optimal design for the EmaxII model (Phi3) %%%
cvx_begin
    cvx_precision high
    variable D_EmaxII_weights(N)
    expression EmaxII_inf_mat(3, 3)
    
    for j = 1:N 
        EmaxII_inf_mat = EmaxII_inf_mat + ...
            D_EmaxII_weights(j)*squeeze(all_z_outer_prods_EmaxII(:, j, :));
    end
    
    % See earier note about log_det vs. det_rootn
    minimize ( -det_rootn(EmaxII_inf_mat))   

    0 <= D_EmaxII_weights <= 1;
    sum(D_EmaxII_weights) == 1;
cvx_end

D_EmaxII_design=[design_points(find(D_EmaxII_weights>tol))' ...
              D_EmaxII_weights(find(D_EmaxII_weights>tol))]'
D_EmaxII_obj = -log(det(EmaxII_inf_mat));

%%% Compute the D-optimal design for the logistic model (Phi4) %%%
cvx_begin
    cvx_precision high
    variable D_logistic_weights(N)
    expression logistic_inf_mat(4, 4)
    
    for j = 1:N 
        logistic_inf_mat = logistic_inf_mat + ...
            D_logistic_weights(j)*squeeze(all_z_outer_prods_logistic(:, j, :));
    end

    % See earier note about log_det vs. det_rootn
    minimize ( -det_rootn(logistic_inf_mat))   

    0 <= D_logistic_weights <= 1;
    sum(D_logistic_weights) == 1;
cvx_end

D_logistic_design=[design_points(find(D_logistic_weights>tol))' ...
              D_logistic_weights(find(D_logistic_weights>tol))]'
D_logistic_obj = -log(det(logistic_inf_mat));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      COMPUTE MAXIMIN OPTIMAL DESIGN              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cvx_begin 
    cvx_precision high 
    variable maximin_weights(N) 
    variable t
    expression lin_inf_mat(2, 2) 
    expression EmaxI_inf_mat(3, 3)
    expression EmaxII_inf_mat(3, 3)
    expression logistic_inf_mat(4, 4)

    for j = 1:N 
        lin_inf_mat = lin_inf_mat + ...
            maximin_weights(j)*squeeze(all_z_outer_prods_linear(:, j, :));
        EmaxI_inf_mat = EmaxI_inf_mat + ...
            maximin_weights(j)*squeeze(all_z_outer_prods_EmaxI(:, j, :));
        EmaxII_inf_mat = EmaxII_inf_mat + ...
            maximin_weights(j)*squeeze(all_z_outer_prods_EmaxII(:, j, :));
        logistic_inf_mat = logistic_inf_mat + ...
            maximin_weights(j)*squeeze(all_z_outer_prods_logistic(:, j, :));
    end

    minimize t 
    
    % Unfortunately, we can't reformulate to det_rootn to avoid successive
    % approximation here because then we will end up with non-convex
    % constraints involving -t*det_rootn(inf_mat)
    -log_det(lin_inf_mat) - D_lin_obj - size(lin_inf_mat, 1)*log(t) <= 0;
    -log_det(EmaxI_inf_mat) - D_EmaxI_obj - size(EmaxI_inf_mat, 1)*log(t) <= 0;
    -log_det(EmaxII_inf_mat) - D_EmaxII_obj - size(EmaxII_inf_mat, 1)*log(t) <= 0;
    -log_det(logistic_inf_mat) - D_logistic_obj - size(logistic_inf_mat, 1)*log(t) <= 0;
    t >= 0; 
    maximin_weights >= 0; 
    sum(maximin_weights) == 1;

cvx_end


maximin_design =[design_points(find(maximin_weights>tol))' ...
              maximin_weights(find(maximin_weights>tol))]'
maximin_obj = cvx_optval;

opt_time = cputime - runningtime


% Calculate efficiencies at the maximin optimal design we found 
linear_D_efficiency = (exp(D_lin_obj)^(1/2))/...
    (exp(-log(det(lin_inf_mat)))^(1/2))
EmaxI_D_efficiency = (exp(D_EmaxI_obj)^(1/3))/...
    (exp(-log(det(EmaxI_inf_mat)))^(1/3))
EmaxII_D_efficiency = (exp(D_EmaxII_obj)^(1/3))/...
    (exp(-log(det(EmaxII_inf_mat)))^(1/3))
logistic_D_efficiency = (exp(D_logistic_obj)^(1/4))/...
    (exp(-log(det(logistic_inf_mat)))^(1/4))

t % optimal value of t

1/t % this should match the smallest of the four efficiencies


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    VERIFY OPTIMALITY OF THE MAXIMIN OPTIMAL DESIGN   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set relaxation parameter for the inequality
tol_lp = 1e-4; 

runningtime=cputime;

% Find Lagrange multipliers using linear programming
d1_vect = zeros(N,1);
d2_vect = zeros(N,1);
d3_vect = zeros(N,1);
d4_vect = zeros(N, 1); 

for j = 1:N
    d1_vect(j) = trace(inv(lin_inf_mat)* ... 
        squeeze(all_z_outer_prods_linear(:, j, :))) - size(lin_inf_mat, 1); 
    d2_vect(j) = trace(inv(EmaxI_inf_mat)* ... 
        squeeze(all_z_outer_prods_EmaxI(:, j, :))) - size(EmaxI_inf_mat, 1); 
    d3_vect(j) = trace(inv(EmaxII_inf_mat)* ... 
        squeeze(all_z_outer_prods_EmaxII(:, j, :))) - size(EmaxII_inf_mat, 1);
    d4_vect(j) = trace(inv(logistic_inf_mat)* ... 
        squeeze(all_z_outer_prods_logistic(:, j, :))) - size(logistic_inf_mat, 1); 
end

D_mat =[d1_vect d2_vect d3_vect d4_vect];
complementarity_mat = diag([-log_det(lin_inf_mat) - D_lin_obj - size(lin_inf_mat, 1)*log(t), ... 
    -log_det(EmaxI_inf_mat) - D_EmaxI_obj - size(EmaxI_inf_mat, 1)*log(t), ... 
    -log_det(EmaxII_inf_mat) - D_EmaxII_obj - size(EmaxII_inf_mat, 1)*log(t), ... 
-log_det(logistic_inf_mat) - D_logistic_obj - size(logistic_inf_mat, 1)*log(t)]);

equality_mat = [size(lin_inf_mat, 1)/t size(EmaxI_inf_mat, 1)/t ... 
    size(EmaxII_inf_mat, 1)/t size(logistic_inf_mat, 1)/t]; 

etas = linprog(ones(4, 1), ... 
    [D_mat' complementarity_mat' complementarity_mat']', tol_lp*ones(N+4*2, 1) , ... 
    equality_mat, 1, ... 
    zeros(4, 1), [])

linprogtime=cputime-runningtime

d_maximin_vect = D_mat*etas;

max(d_maximin_vect) % should be around tol_lp
abs(max(d_maximin_vect) - tol_lp) % Should be very very small 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   MAKE PLOTS                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

set(groot,'defaultLineLineWidth',1.2)


% FIGURE 2
tiledlayout(2,3)
zero_line = zeros(N, 1);

ax1=nexttile;
plot(design_points,d1_vect,design_points,zero_line,'r--')
xlim([0 500])
title(ax1,'(a) $\phi_1$-optimality', 'Interpreter', 'latex', ...
        'FontSize', 9)
xlabel('$u_i$', 'Interpreter', 'latex', ...
        'FontSize', 12)
ylabel('$d_{\phi_1, f_1}(u_i, {\bf w}^{*mm}$)', 'Interpreter', 'latex', ...
        'FontSize', 9)
set(gca,'linewidth',1.2)
    
ax2=nexttile;
plot(design_points,d2_vect,design_points,zero_line,'r--')
xlim([0 500])
title(ax2,'(b) $\phi_2$-optimality', 'Interpreter', 'latex', ...
        'FontSize', 9)
xlabel('$u_i$', 'Interpreter', 'latex', ...
        'FontSize', 12)
ylabel('$d_{\phi_2, f_2}(u_i, {\bf w}^{*mm}$)', 'Interpreter', 'latex', ...
        'FontSize', 9)
set(gca,'linewidth',1.2)

    
ax3=nexttile;
plot(design_points,d3_vect,design_points,zero_line,'r--')
xlim([0 500])
title(ax3,'(c) $\phi_3$-optimality', 'Interpreter', 'latex', ...
        'FontSize', 9)
xlabel('$u_i$', 'Interpreter', 'latex', ...
        'FontSize', 12)
ylabel('$d_{\phi_3, f_3}(u_i, {\bf w}^{*mm}$)', 'Interpreter', 'latex', ...
        'FontSize', 9)
set(gca,'linewidth',1.2)


ax4=nexttile;
plot(design_points,d4_vect,design_points,zero_line,'r--')
xlim([0 500])
title(ax4,'(d) $\phi_4$-optimality', 'Interpreter', 'latex', ...
        'FontSize', 9)
xlabel('$u_i$', 'Interpreter', 'latex', ...
        'FontSize', 12)
ylabel('$d_{\phi_4, f_4}(u_i, {\bf w}^{*mm}$)', 'Interpreter', 'latex', ...
        'FontSize', 9)
set(gca,'linewidth',1.2)

    


delta_line = tol_lp*ones(N, 1);

ax5=nexttile;
plot(design_points,d_maximin_vect,design_points,delta_line,'r--')
xlim([0 500])
title(ax5,{['(e) Multi-objective optimality']}, 'Interpreter', 'latex', ...
        'FontSize', 9)
xlabel('$u_i$', 'Interpreter', 'latex', ...
        'FontSize', 12)
ylabel('$\sum_{k=1}^4 \eta_k^* d_{\phi_k, f_k}(u_i, {\bf w}^{*mm})$', ...
        'Interpreter','latex','FontSize',9)
limsy=get(gca,'YLim');
ylim([limsy(1) 0.05])
set(gca,'linewidth',1.2)


h = gcf;
set(h, 'PaperUnits','centimeters');
set(h, 'Units','centimeters');

pos=get(h,'Position');
set(h, 'Position', [pos(1),pos(2), 20, 8]);

pos=get(h,'Position');
set(h, 'PaperSize', [pos(3) pos(4)]);
set(h, 'PaperPositionMode', 'manual');
set(h, 'PaperPosition',[0 0 pos(3) pos(4)]);

print('Figure2', '-dpdf')

