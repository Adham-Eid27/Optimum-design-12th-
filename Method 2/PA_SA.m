% Hybrid PSO-SA vs Standard PSO PID Optimization Comparison
% Plant: Gp(s) = 7.84 / (3s^2 + 5.04s + 7.84)
clc; clear; close all;
 
%% DC Motor Transfer Function
num = 7.84;
den = [3, 5.04, 7.84];
motor_tf = tf(num, den);
 
%% Common Parameters
n_particles = 30;      % Number of particles
max_iter = 50;         % Maximum iterations
Kp_range = [0, 50];    % Range for Kp
Ki_range = [0, 50];    % Range for Ki
Kd_range = [0, 10];    % Range for Kd
 
%% 1. First run Standard PSO Optimization
disp('Running Standard PSO Optimization...');
w = 0.7;               % Inertia weight
c1 = 1.5;              % Cognitive coefficient
c2 = 1.5;              % Social coefficient
 
% Initialize particles
particles = struct('position', [], 'velocity', [], 'cost', [], 'best_position', [], 'best_cost', []);
pso_global_best.cost = inf;
pso_global_best.position = [];
 
for i = 1:n_particles
    Kp = Kp_range(1) + (Kp_range(2)-Kp_range(1))*rand();
    Ki = Ki_range(1) + (Ki_range(2)-Ki_range(1))*rand();
    Kd = Kd_range(1) + (Kd_range(2)-Kd_range(1))*rand();
    
    particles(i).position = [Kp, Ki, Kd];
    particles(i).velocity = zeros(1, 3);
    
    [cost, ~, ~, ~] = evaluate_pid([Kp, Ki, Kd], motor_tf);
    particles(i).cost = cost;
    particles(i).best_position = particles(i).position;
    particles(i).best_cost = cost;
    
    if cost < pso_global_best.cost
        pso_global_best.cost = cost;
        pso_global_best.position = particles(i).position;
    end
end
 
% PSO Optimization Loop
for iter = 1:max_iter
    for i = 1:n_particles
        r1 = rand(1,3);
        r2 = rand(1,3);
        particles(i).velocity = w * particles(i).velocity + ...
                              c1 * r1 .* (particles(i).best_position - particles(i).position) + ...
                              c2 * r2 .* (pso_global_best.position - particles(i).position);
        
        particles(i).position = particles(i).position + particles(i).velocity;
        
        % Apply bounds
        particles(i).position = max([Kp_range(1), Ki_range(1), Kd_range(1)], ...
                                   min([Kp_range(2), Ki_range(2), Kd_range(2)], particles(i).position));
        
        [cost, ~, ~, ~] = evaluate_pid(particles(i).position, motor_tf);
        
        if cost < particles(i).best_cost
            particles(i).best_cost = cost;
            particles(i).best_position = particles(i).position;
            
            if cost < pso_global_best.cost
                pso_global_best.cost = cost;
                pso_global_best.position = particles(i).position;
            end
        end
    end
    w = w * 0.98; % Inertia weight decay
end
 
% Get final PSO results
[~, pso_step_info, pso_sys_cl, pso_ss_error] = evaluate_pid(pso_global_best.position, motor_tf);
 
%% 2. Then run Hybrid PSO-SA Optimization
disp('Running Hybrid PSO-SA Optimization...');
w = 0.7;               % Reset inertia weight
c1 = 1.5;              % Cognitive coefficient
c2 = 1.5;              % Social coefficient
 
% SA Parameters
T0 = 100;              % Initial temperature
alpha = 0.95;          % Cooling rate
sa_iter = 5;           % SA iterations per PSO iteration
 
% Initialize particles (same initialization as PSO)
for i = 1:n_particles
    Kp = Kp_range(1) + (Kp_range(2)-Kp_range(1))*rand();
    Ki = Ki_range(1) + (Ki_range(2)-Ki_range(1))*rand();
    Kd = Kd_range(1) + (Kd_range(2)-Kd_range(1))*rand();
    
    particles(i).position = [Kp, Ki, Kd];
    particles(i).velocity = zeros(1, 3);
    
    [cost, ~, ~, ~] = evaluate_pid([Kp, Ki, Kd], motor_tf);
    particles(i).cost = cost;
    particles(i).best_position = particles(i).position;
    particles(i).best_cost = cost;
    
    if cost < pso_global_best.cost
        pso_global_best.cost = cost;
        pso_global_best.position = particles(i).position;
    end
end
 
% Hybrid PSO-SA Optimization Loop
T = T0;  % Initial temperature
for iter = 1:max_iter
    % Standard PSO Update
    for i = 1:n_particles
        r1 = rand(1,3);
        r2 = rand(1,3);
        particles(i).velocity = w * particles(i).velocity + ...
                              c1 * r1 .* (particles(i).best_position - particles(i).position) + ...
                              c2 * r2 .* (pso_global_best.position - particles(i).position);
        
        particles(i).position = particles(i).position + particles(i).velocity;
        particles(i).position = max([Kp_range(1), Ki_range(1), Kd_range(1)], ...
                               min([Kp_range(2), Ki_range(2), Kd_range(2)], particles(i).position));
        
        [cost, ~, ~, ~] = evaluate_pid(particles(i).position, motor_tf);
        
        if cost < particles(i).best_cost
            particles(i).best_cost = cost;
            particles(i).best_position = particles(i).position;
            
            if cost < pso_global_best.cost
                pso_global_best.cost = cost;
                pso_global_best.position = particles(i).position;
            end
        end
    end
    
    % Simulated Annealing Local Search
    for k = 1:sa_iter
        for i = 1:n_particles
            neighbor = particles(i).best_position + T/T0*randn(1,3).*(Kp_range(2)-Kp_range(1))/10;
            neighbor = max([Kp_range(1), Ki_range(1), Kd_range(1)], ...
                         min([Kp_range(2), Ki_range(2), Kd_range(2)], neighbor));
            
            [neighbor_cost, ~, ~, ~] = evaluate_pid(neighbor, motor_tf);
            
            delta_cost = neighbor_cost - particles(i).best_cost;
            if delta_cost < 0 || rand() < exp(-delta_cost/T)
                particles(i).best_position = neighbor;
                particles(i).best_cost = neighbor_cost;
                
                if neighbor_cost < pso_global_best.cost
                    pso_global_best.cost = neighbor_cost;
                    pso_global_best.position = neighbor;
                end
            end
        end
    end
    
    T = alpha * T; % Cool down temperature
    w = w * 0.98;  % Inertia weight decay
end
 
% Get final PSO-SA results
[~, pso_sa_step_info, pso_sa_sys_cl, pso_sa_ss_error] = evaluate_pid(pso_global_best.position, motor_tf);
 
%% Display Comparison Results
% Scale open-loop to settle at 1
DC_gain = dcgain(motor_tf);
scaled_open_loop = motor_tf/DC_gain;

% Get open-loop response metrics
[y_open, t_open] = step(scaled_open_loop);
open_info = stepinfo(y_open, t_open);
ss_error_open = abs(1 - y_open(end)) * 100;

% Display all results
fprintf('\n=== Complete Performance Comparison ===\n');

fprintf('\n--- Scaled Open-Loop (no PID) ---\n');
fprintf('Rise Time:    %.4f s\n', open_info.RiseTime);
fprintf('Settling Time:%.4f s\n', open_info.SettlingTime);
fprintf('Overshoot:    %.2f%%\n', open_info.Overshoot);

fprintf('\n--- Standard PSO Results ---\n');
fprintf('PID Gains: Kp = %.4f, Ki = %.4f, Kd = %.4f\n', pso_global_best.position);
fprintf('Rise Time:    %.4f s\n', pso_step_info.RiseTime);
fprintf('Settling Time:%.4f s\n', pso_step_info.SettlingTime);
fprintf('Overshoot:    %.2f%%\n', pso_step_info.Overshoot);

 
fprintf('\n--- Hybrid PSO-SA Results ---\n');
fprintf('PID Gains: Kp = %.4f, Ki = %.4f, Kd = %.4f\n', pso_global_best.position);
fprintf('Rise Time:    %.4f s\n', pso_sa_step_info.RiseTime);
fprintf('Settling Time:%.4f s\n', pso_sa_step_info.SettlingTime);
fprintf('Overshoot:    %.2f%%\n', pso_sa_step_info.Overshoot);

 
%% Triple Comparison Plot
figure;
set(gcf, 'Position', [100, 100, 1000, 700]);

% Simulate all responses with consistent time vector
t = 0:0.01:10;
[y_open_scaled, ~] = step(scaled_open_loop, t);
[y_pso, ~] = step(pso_sys_cl, t);
[y_pso_sa, ~] = step(pso_sa_sys_cl, t);

% Plot all responses
plot(t, y_open_scaled, 'Color', [0 0.5 0], 'LineWidth', 2); % Dark green for open-loop
hold on;
plot(t, y_pso, 'b', 'LineWidth', 2); % Blue for PSO
plot(t, y_pso_sa, 'r', 'LineWidth', 2); % Red for PSO-SA
plot([0 t(end)], [1 1], 'k--', 'LineWidth', 1); % Reference line
hold off;

% Format plot
title('System Response Comparison: Open-Loop vs PSO vs PSO-SA PID', 'FontSize', 16);
xlabel('Time (seconds)', 'FontSize', 14);
ylabel('Amplitude', 'FontSize', 14);
legend('Scaled Open-Loop', 'Standard PSO PID', 'Hybrid PSO-SA PID', 'Reference', 'Location', 'southeast');
grid on;
xlim([0 10]);
ylim([0 1.2]);
%% Cost Function with Penalty Method
function [cost, step_info, sys_cl, steady_state_error] = evaluate_pid(pid_params, motor_tf)
    Kp = pid_params(1);
    Ki = pid_params(2);
    Kd = pid_params(3);
    
    % Create PID controller with filter on derivative term
    N = 100;  % Filter coefficient for derivative term
    pid_tf = pid(Kp, Ki, Kd, N);
    
    % Closed-loop system
    sys_cl = feedback(pid_tf * motor_tf, 1);
    
    % Get step response information with extended simulation
    try
        t = 0:0.01:10;  % Extended simulation time
        [y, t] = step(sys_cl, t);
        
        % Calculate step info with relaxed settling threshold
        step_info = stepinfo(y, t, 'SettlingTimeThreshold', 0.05);
        
        % Handle NaN settling time
        if isnan(step_info.SettlingTime)
            step_info.SettlingTime = 10;  % Max simulation time
        end
        
        % Calculate steady-state error
        steady_state_value = y(end);
        steady_state_error = abs(1 - steady_state_value) * 100;
        
        % Extract performance metrics
        overshoot = step_info.Overshoot;
        if isempty(overshoot)
            overshoot = 0;
        end
        
        rise_time = step_info.RiseTime;
        settling_time = step_info.SettlingTime;
        
        % Objective: Minimize overshoot
        base_cost = overshoot;
        
        % Constraints with penalty functions
        penalty = 0;
        
        % Steady-state error < 5%
        if steady_state_error >= 5
            penalty = penalty + 1000 + (steady_state_error - 5)^2;
        end
        
        % Settling time between 1 and 1.5 sec
        if settling_time < 1
            penalty = penalty + 1000 + (1 - settling_time)^2;
        elseif settling_time > 1.5
            penalty = penalty + 1000 + (settling_time - 1.5)^2;
        end
        
        % Rise time at least 0.5 sec
        if rise_time < 0.5
            penalty = penalty + 1000 + (0.5 - rise_time)^2;
        end
        
        % Additional penalty for non-settling systems
        if step_info.SettlingTime >= 10
            penalty = penalty + 2000;
        end
        
        % Total cost
        cost = base_cost + penalty;
        
    catch
        % If simulation fails (unstable system), assign high cost
        cost = 1e6;
        step_info = struct('RiseTime', 0, 'SettlingTime', 100, 'Overshoot', 100);
        steady_state_error = 100;
        sys_cl = tf(1,[1 1]);
    end
end