%% PID Tuning with GA (Minimize Overshoot)

% 1. Define plant transfer function
num = 7.84;
den = [3, 5.04, 7.84];
Gp = tf(num, den);

% 2. GA options & bounds
nVars   = 3;                % [Kp, Ki, Kd]
lb      = [0,   0,   0];    % lower bounds
ub      = [100, 100, 10];   % upper bounds

options = optimoptions('ga', ...
    'Display', 'iter', ...
    'PopulationSize', 800, ...     
    'MaxGenerations', 300, ...     
    'EliteCount', 8, ...           
    'CrossoverFraction', 0.8, ...  
    'MutationFcn', {@mutationadaptfeasible}, ...  
    'UseParallel', true);

% 3. Run GA to tune PID
tic;    % start timer
[x_opt, ~] = ga( ...
    @(x) pidObjective(x, Gp), ...
    nVars, [], [], [], [], lb, ub, ...
    @(x) pidConstraints(x, Gp), ...
    options);
elapsedGA = toc;  % stop timer and store elapsed time

fprintf('GA optimization took %.2f seconds.\n\n', elapsedGA);

Kp_opt = x_opt(1);
Ki_opt = x_opt(2);
Kd_opt = x_opt(3);

fprintf('\nOptimized PID gains (minimize overshoot, Rt ≥ 0.5s):\n');
fprintf('  Kp = %.4f\n', Kp_opt);
fprintf('  Ki = %.4f\n', Ki_opt);
fprintf('  Kd = %.4f\n\n', Kd_opt);

% 4. Prepare responses for plotting
t      = 0:0.01:10;
[y_ol, ~] = step(Gp, t);
C_opt     = pid(Kp_opt, Ki_opt, Kd_opt);
sys_cl    = feedback(C_opt*Gp,1);
[y_cl, ~] = step(sys_cl, t);
ess       = abs(1 - y_cl(end));  % Steady-state error

% 5. Plot comparison
figure;
plot(t, y_ol,  'LineWidth',1.5); hold on;
plot(t, y_cl, '--','LineWidth',1.5); hold off;
grid on; xlabel('Time (s)'); ylabel('Output');
title('Open-Loop vs. GA-Tuned PID Closed-Loop');
legend('Open-Loop','Closed-Loop (PID)','Location','Best');

% 6. Print performance metrics
info_ol = stepinfo(Gp);
info_cl = stepinfo(sys_cl);                 

fprintf('--- Scaled Open-Loop (no PID) ---\n');
fprintf(' Rise Time:     %.4f s\n', info_ol.RiseTime);
fprintf(' Settling Time: %.4f s\n', info_ol.SettlingTime);
fprintf(' Overshoot:     %.2f%%\n\n', info_ol.Overshoot);

fprintf('--- Closed-Loop (with PID) ---\n');
fprintf(' Rise Time:     %.4f s\n', info_cl.RiseTime);
fprintf(' Settling Time: %.4f s\n', info_cl.SettlingTime);
fprintf(' Overshoot:     %.2f%%\n', info_cl.Overshoot);
fprintf(' Steady-State Error (ess): %.4f\n', ess);


%% --- Objective function: minimize overshoot (penalize infeasible) ---
function cost = pidObjective(x, Gp)
    C   = pid(x(1), x(2), x(3));
    sys = feedback(C*Gp,1);
    info = stepinfo(sys);
    if isempty(info) || isnan(info.Overshoot) || isinf(info.Overshoot)
        cost = 1e6;  % penalty if system is unstable or info is bad
    else
        cost = info.Overshoot;
    end
end

%% --- Constraints:
function [c, ceq] = pidConstraints(x, Gp)
    C   = pid(x(1), x(2), x(3));
    sys = feedback(C*Gp,1);
    info = stepinfo(sys);

    [y, ~] = step(sys, 0:0.01:10);
    ess = abs(1 - y(end));  % steady-state error

    c1 = ess - 0.05;             % ess ≤ 0.05
    c2 = info.SettlingTime - 1.5;% Ts ≤ 1.5
    c3 = 1.2 - info.SettlingTime;% Ts ≥ 1.2
    c4 = 0.5 - info.RiseTime;    % Rt ≥ 0.5
    
    %check if any response returns incorrent data
    if isempty(info) || any(isnan([info.SettlingTime, info.RiseTime]))
        c = [1; 1; 1; 1]; 
    else
        c = [c1; c2; c3; c4];
    end
    ceq = [];
end
