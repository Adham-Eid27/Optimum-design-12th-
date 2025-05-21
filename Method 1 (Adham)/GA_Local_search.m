%% PID Tuning with GA + fmincon Hybrid (Minimize Overshoot)

% 1. Define plant transfer function
num = 7.84;
den = [3, 5.04, 7.84];
Gp = tf(num, den);

% 2. Define bounds for PID parameters
nVars   = 3;                  
lb      = [0,   0,   0];      
ub      = [100, 100, 10];     

% 3. fmincon options (used as HybridFcn)
fminconOptions = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'sqp', ...
    'MaxIterations', 100, ...
    'OptimalityTolerance', 1e-6);

% 4. GA options
options = optimoptions('ga', ...
    'Display', 'iter', ...
    'PopulationSize', 800, ...     
    'MaxGenerations', 300, ...     
    'EliteCount', 8, ...           
    'CrossoverFraction', 0.8, ...  
    'MutationFcn', {@mutationadaptfeasible}, ...  
    'UseParallel', true, ...
    'PlotFcn', {@gaplotbestf, @gaplotscorediversity}, ...
    'HybridFcn', {@fmincon, fminconOptions});  % Hybrid optimization

% 5. Run GA + fmincon Hybrid Optimization
tic;
[x_opt, ~] = ga( ...
    @(x) pidObjective(x, Gp), ...
    nVars, [], [], [], [], lb, ub, ...
    @(x) pidConstraints(x, Gp), ...
    options);
elapsedGA = toc;

Kp_opt = x_opt(1);
Ki_opt = x_opt(2);
Kd_opt = x_opt(3);

% 6. Simulation Time Vector
t = 0:0.01:10;

% 7. Open-Loop Response
[y_ol, ~] = step(Gp, t);
info_ol = stepinfo(Gp);
Kp_ol = dcgain(Gp);
ess_ol = 1 / (1 + Kp_ol);

% 8. GA-only Response
C_ga = pid(Kp_opt, Ki_opt, Kd_opt);
sys_ga = feedback(C_ga * Gp, 1);
[y_ga, ~] = step(sys_ga, t);
info_ga = stepinfo(sys_ga);
ess_ga = abs(1 - y_ga(end));

% 9. fmincon Refinement (manually refine again for better separation)
[x_refined, ~] = fmincon(@(x) pidObjective(x, Gp), x_opt, ...
    [], [], [], [], lb, ub, @(x) pidConstraints(x, Gp), fminconOptions);

Kp_ref = x_refined(1);
Ki_ref = x_refined(2);
Kd_ref = x_refined(3);

C_ref = pid(Kp_ref, Ki_ref, Kd_ref);
sys_ref = feedback(C_ref * Gp, 1);
[y_ref, ~] = step(sys_ref, t);
info_ref = stepinfo(sys_ref);
ess_ref = abs(1 - y_ref(end));

% 10. Plot all responses
figure;
plot(t, y_ol,  'k-',  'LineWidth', 1.5); hold on;
plot(t, y_ga,  'b--', 'LineWidth', 1.5);
plot(t, y_ref, 'r-.', 'LineWidth', 1.5);
grid on; xlabel('Time (s)'); ylabel('Output');
title('Step Responses: Open-Loop vs GA vs GA+fmincon');
legend('Open-Loop','GA PID','GA+fmincon PID','Location','Best');

% 11. Print Results
fprintf('\n==============================\n');
fprintf('--- Open-Loop (no PID) ---\n');
fprintf(' Rise Time:     %.4f s\n', info_ol.RiseTime);
fprintf(' Settling Time: %.4f s\n', info_ol.SettlingTime);
fprintf(' Overshoot:     %.2f%%\n', info_ol.Overshoot);
fprintf(' Steady-State Error: %.2f%%\n', ess_ol * 100);

fprintf('\n--- GA-Only PID ---\n');
fprintf(' Kp = %.4f, Ki = %.4f, Kd = %.4f\n', Kp_opt, Ki_opt, Kd_opt);
fprintf(' Rise Time:     %.4f s\n', info_ga.RiseTime);
fprintf(' Settling Time: %.4f s\n', info_ga.SettlingTime);
fprintf(' Overshoot:     %.2f%%\n', info_ga.Overshoot);
fprintf(' Steady-State Error: %.2f%%\n', ess_ga * 100);

fprintf('\n--- GA + fmincon Hybrid PID ---\n');
fprintf(' Kp = %.4f, Ki = %.4f, Kd = %.4f\n', Kp_ref, Ki_ref, Kd_ref);
fprintf(' Rise Time:     %.4f s\n', info_ref.RiseTime);
fprintf(' Settling Time: %.4f s\n', info_ref.SettlingTime);
fprintf(' Overshoot:     %.2f%%\n', info_ref.Overshoot);
fprintf(' Steady-State Error: %.2f%%\n', ess_ref * 100);
fprintf('==============================\n');


%% --- Objective Function ---
function cost = pidObjective(x, Gp)
    C   = pid(x(1), x(2), x(3));
    sys = feedback(C * Gp, 1);
    info = stepinfo(sys);
    if isempty(info) || isnan(info.Overshoot) || isinf(info.Overshoot)
        cost = 1e6;
    else
        cost = info.Overshoot;
    end
end

%% --- Constraint Function ---
function [c, ceq] = pidConstraints(x, Gp)
    C = pid(x(1), x(2), x(3));
    sys = feedback(C * Gp, 1);
    info = stepinfo(sys);

    [y, ~] = step(sys, 0:0.01:10);
    ess = abs(1 - y(end));

    c1 = ess - 0.05;               % ess ≤ 5%
    c2 = info.SettlingTime - 1.5;  % Ts ≤ 1.5
    c3 = 1 - info.SettlingTime;    % Ts ≥ 1
    c4 = 0.5 - info.RiseTime;      % Rt ≥ 0.5

    if isempty(info) || any(isnan([info.SettlingTime, info.RiseTime]))
        c = [1; 1; 1; 1]; 
    else
        c = [c1; c2; c3; c4];
    end
    ceq = [];
end
