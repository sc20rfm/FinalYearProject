close all;
clear all;

% Load the robot model
robot = loadrobot('kinovaGen3', 'DataFormat', 'row', 'Gravity', [0 0 -9.81]);
endEffector = "EndEffector_Link";

% Initial configuration of the robot
startJointConfig = [0 0.1 0.0 0.2 -0.0 1.0 0.0]';
q(:, 1) = startJointConfig;

% Define the 3D target position for the end-effector
targetPosition = [0.7 0.1 0.3]'; % Example target position
numIterations = 200; % Number of iterations for the trajectory

% Preallocate arrays for storing joint configurations
jointTrajectory = zeros(7, numIterations + 1);
jointTrajectory(:, 1) = startJointConfig;

% Inverse kinematics loop to move the end-effector to the target position
for n = 1:numIterations
    curJointConfig = jointTrajectory(:, n);
    J_temp = geometricJacobian(robot, curJointConfig', endEffector);
    J = J_temp(4:6, :); % Jacobian for Cartesian velocity
    
    Htmp = getTransform(robot, curJointConfig', endEffector); % Forward kinematics
    endEffectorPos = Htmp(1:3, 4); % End-effector position
    error = (targetPosition - endEffectorPos); % Positional error
    
    % Break the loop if the error is small enough
    if norm(error) < 1e-3
        jointTrajectory = jointTrajectory(:, 1:n); % Trim unused entries
        break;
    end
    
    dq = 0.01 * pinv(J) * error; % Change in joint angles using pseudoinverse of Jacobian
    jointTrajectory(:, n + 1) = curJointConfig + dq; % Update joint configuration
    
    % Visualization
    if mod(n, 5) == 0
        show(robot, jointTrajectory(:, n + 1)', 'PreservePlot', false, 'Frames', 'off');
        axis([-1 1 -1 1 -0.1 1.5]);
        hold on;
        plot3(Htmp(1, 4), Htmp(2, 4), Htmp(3, 4), 'r.', 'MarkerSize', 10); % Plot end-effector position
        drawnow;
    end
end

% Plot the joint angles over iterations
figure;
for i = 1:7
    subplot(7, 1, i);
    plot(0:(size(jointTrajectory, 2) - 1), jointTrajectory(i, :), 'b', 'LineWidth', 2);
    axis tight;
    legend(['Joint ' num2str(i) ' Angle']);
    xlabel('Iteration');
    ylabel('Angle (rad)');
end

% Save the joint trajectory to a file
save('jointTrajectory.mat', 'jointTrajectory');
