% Standard PSO-PID Optimization for DC Motor Control
% Plant: Gp(s) = 0.067 / (0.00113s^2 + 0.0078854s + 0.0171)
clc;
clear;
close all;
 
%% DC Motor Transfer Function
num = 0.067;
den = [0.00113, 0.0078854, 0.0171];
motor_tf = tf(num, den);
 
%% PSO Parameters
n_particles = 30;      % Number of particles
max_iter = 100;         % Maximum iterations
w = 0.9;               % Inertia weight
c1 = 1.5;              % Cognitive coefficient
c2 = 1.5;              % Social coefficient
Kp_range = [0, 50];    % Range for Kp
Ki_range = [0, 50];    % Range for Ki
Kd_range = [0, 10];    % Range for Kd
 
%% Initialize particles
particles = struct('position', [], 'velocity', [], 'cost', [], 'best_position', [], 'best_cost', []);
global_best.cost = inf;
global_best.position = [];
 
for i = 1:n_particles
    % Random initialization within bounds
    Kp = Kp_range(1) + (Kp_range(2)-Kp_range(1))*rand();
    Ki = Ki_range(1) + (Ki_range(2)-Ki_range(1))*rand();
    Kd = Kd_range(1) + (Kd_range(2)-Kd_range(1))*rand();
    
    particles(i).position = [Kp, Ki, Kd];
    particles(i).velocity = zeros(1, 3);
    
    % Evaluate initial cost
    [cost, ~, ~, steady_state_error] = evaluate_pid([Kp, Ki, Kd], motor_tf);
    particles(i).cost = cost;
    particles(i).best_position = particles(i).position;
    particles(i).best_cost = cost;
    
    % Update global best
    if cost < global_best.cost
        global_best.cost = cost;
        global_best.position = particles(i).position;
    end
end
 
%% PSO Optimization Loop
for iter = 1:max_iter
    for i = 1:n_particles
        % Update velocity
        r1 = rand(1,3);
        r2 = rand(1,3);
        
        cognitive = c1 * r1 .* (particles(i).best_position - particles(i).position);
        social = c2 * r2 .* (global_best.position - particles(i).position);
        
        particles(i).velocity = w * particles(i).velocity + cognitive + social;
        
        % Update position
        particles(i).position = particles(i).position + particles(i).velocity;
        
        % Apply bounds
        particles(i).position(1) = max(Kp_range(1), min(Kp_range(2), particles(i).position(1)));
        particles(i).position(2) = max(Ki_range(1), min(Ki_range(2), particles(i).position(2)));
        particles(i).position(3) = max(Kd_range(1), min(Kd_range(2), particles(i).position(3)));
        
        % Evaluate new position
        [cost, step_info, ~, steady_state_error] = evaluate_pid(particles(i).position, motor_tf);
        
        % Update personal best
        if cost < particles(i).best_cost
            particles(i).best_cost = cost;
            particles(i).best_position = particles(i).position;
            
            % Update global best
            if cost < global_best.cost
                global_best.cost = cost;
                global_best.position = particles(i).position;
                best_step_info = step_info;
                best_steady_state_error = steady_state_error;
            end
        end
    end
    
    % Display progress
    fprintf('Iteration %d: Global Best score = %.4f, Kp=%.3f, Ki=%.3f, Kd=%.3f\n', ...
            iter, global_best.cost, global_best.position(1), global_best.position(2), global_best.position(3));
    
    % Optional: Adaptive inertia weight
    w = w * 0.99;
end
 
%% Display Results
fprintf('\nOptimization Results:\n');
fprintf('Best PID Parameters: Kp = %.3f, Ki = %.3f, Kd = %.3f\n', ...
        global_best.position(1), global_best.position(2), global_best.position(3));
 
% Get open-loop response metrics
[y_open, t_open] = step(motor_tf);
open_info = stepinfo(y_open, t_open);
ss_error_open = abs(1 - y_open(end)) * 100;
 
fprintf('\n--- Scaled Open-Loop (no PID) ---\n');
fprintf('Rise Time:    %.4f s\n', open_info.RiseTime);
fprintf('Settling Time:%.4f s\n', open_info.SettlingTime);
fprintf('Overshoot:    %.2f%%\n', open_info.Overshoot);
fprintf('Steady-State Error: %.2f%%\n', ss_error_open);
 
fprintf('\n--- Closed-Loop (with PID) ---\n');
fprintf('Rise Time:    %.4f s\n', best_step_info.RiseTime);
fprintf('Settling Time:%.4f s\n', best_step_info.SettlingTime);
fprintf('Overshoot:    %.2f%%\n', best_step_info.Overshoot);
fprintf('Steady-State Error: %.2f%%\n', best_steady_state_error);
 
% ... (previous code remains the same until the comparative plot section)
 
%% Comparative Plot: Open-Loop vs PID-Controlled
[~, ~, sys_cl] = evaluate_pid(global_best.position, motor_tf);
 
% Calculate DC gain of open-loop system
DC_gain = dcgain(motor_tf);
 
% Create scaled open-loop system that settles at 1
scaled_open_loop = motor_tf/DC_gain;
 
figure;
set(gcf, 'Position', [100, 100, 800, 500]);
 
% Simulate responses with consistent time vector
t = 0:0.001:2;
[y_open, t_open] = step(scaled_open_loop, t);  % Using scaled open-loop
[y_pid, t_pid] = step(sys_cl, t);
 
% Plot both responses
plot(t_open, y_open, 'b', 'LineWidth', 2);
hold on;
plot(t_pid, y_pid, 'r', 'LineWidth', 2);
plot([0 t(end)], [1 1], 'k--', 'LineWidth', 1); % Reference line
hold off;
 
% Add labels and title
title('System Response Comparison: Open-Loop vs PSO-Optimized PID', 'FontSize', 14);
xlabel('Time (seconds)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
legend('Scaled Open-Loop Response', 'PSO-Optimized PID', 'Reference', 'Location', 'southeast');
grid on;
 
% Set consistent axes
xlim([0 2]);
ylim([0 max(1.2, 1.2*max([y_open; y_pid]))]);
 
% Recalculate open-loop metrics with scaled system
[y_open_scaled, t_open_scaled] = step(scaled_open_loop);
open_info_scaled = stepinfo(y_open_scaled, t_open_scaled);
ss_error_open_scaled = abs(1 - y_open_scaled(end)) * 100;
 
%% Cost Function with Penalty Method
function [cost, step_info, sys_cl, steady_state_error] = evaluate_pid(pid_params, motor_tf)
    Kp = pid_params(1);
    Ki = pid_params(2);
    Kd = pid_params(3);
    
    % Create PID controller
    pid_tf = pid(Kp, Ki, Kd);
    
    % Closed-loop system
    sys_cl = feedback(pid_tf * motor_tf, 1);
    
    % Get step response information
    try
        [y, t] = step(sys_cl);
        step_info = stepinfo(sys_cl);
        
        % Calculate steady-state error properly
        steady_state_value = y(end);
        steady_state_error = abs(1 - steady_state_value) * 100; % in percentage
        
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
        
        % Total cost
        cost = base_cost + penalty;
        
    catch
        % If simulation fails (unstable system), assign high cost
        cost = 1e6;
        step_info.Overshoot = 100;
        step_info.RiseTime = 0;
        step_info.SettlingTime = 100;
        steady_state_error = 100;
        sys_cl = tf(1,[1 1]); % Dummy stable system
    end
end
