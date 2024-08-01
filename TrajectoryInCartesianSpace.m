close all;
clear all;

% Load robot model
robot = loadrobot('kinovaGen3', 'DataFormat', 'row', 'Gravity', [0 0 -9.81]);
endEffector = "EndEffector_Link";

% Initial configuration of the robot
startJointConfig = [0 0.1 0.0 0.2 -0.0 1.0 0.0]';
q(:,1) = startJointConfig;

% Parameters for a larger sinusoidal trajectory
numPoints = 100;
time = linspace(0, 10, numPoints);
x_coord = linspace(-4, 4, numPoints); % Increase the range of x values
y_coord = 2 * sin(x_coord); % Increase the amplitude of y values
z_coord = 0.5 * ones(1, numPoints); % Constant z value

% Create the sinusoidal trajectory in a 4-column array
trajectory = [time', x_coord', y_coord', z_coord'];

% Create the target points for the robot to follow
for i = 1:numPoints
    target(:,i) = [0.4; trajectory(i,2)*0.2; trajectory(i,3)*0.2 + 0.5]; % Adjust scaling factor
end

% Jacobian-based inverse kinematics
for n = 1:numPoints
    curJointConfig = q(:,n);
    J_temp = geometricJacobian(robot, curJointConfig', endEffector);
    J = J_temp(4:6,:); % Jacobian for Cartesian velocity
    
    Htmp = getTransform(robot, curJointConfig', endEffector); % Forward kinematics
    end_effector_pos(:,n) = Htmp(1:3,4); % End-effector position
    e = (target(:,n) - end_effector_pos(:,n));
    
    dq = 0.6 * pinv(J) * e;
    q(:,n+1) = q(:,n) + dq;
    
    if mod(n,2) == 0 && n > 10
        show(robot, q(:,n+1)', 'PreservePlot', false, 'Frames', 'off'); % Show robot at the updated configuration       
        axis([-1 1 -1 1 -0.1 1.5]);
        hold on
        plot3(Htmp(1,4), Htmp(2,4), Htmp(3,4), 'r.', 'MarkerSize', 10); % Plot end-effector position
        set(gca, 'FontSize', 20);
        drawnow;
    end
end

% Plot reference and observed position
figure
for n = 1:3
    subplot(3,1,n)
    plot(target(n,:), 'r', 'LineWidth', 1);
    hold on
    plot(end_effector_pos(n,:), 'b', 'LineWidth', 2);
    axis tight
    legend('target', 'planned');
end
