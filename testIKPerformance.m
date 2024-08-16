function testIKPerformance()
    close all;
    clear all;
    
    % Load the robot model
    robot = loadrobot('kinovaGen3', 'DataFormat', 'row', 'Gravity', [0 0 -9.81]);
    endEffector = "EndEffector_Link";
    
    % Initial configuration of the robot
    startJointConfig = [0 0.1 0.0 0.2 -0.0 1.0 0.0]';
    q(:, 1) = startJointConfig;
    
    % Define the 3D target position for the end-effector
    targetPosition = [0.7 0.1 0.3]';
    numIterations = 2000; % Maximum number of iterations
    initialStepSize = 0.01; % Initial step size
    initialLambda = 0.1; % Initial Levenberg-Marquardt damping factor
    minErrorThreshold = 0.0001; % Error threshold for stopping (0.1 mm)
    momentumFactor = 0.9; % Momentum factor
    patience = 100; % Early stopping patience
    minImprovement = 1e-6; % Minimum improvement to reset patience
    
    % Preallocate arrays for storing joint configurations and momentum
    jointTrajectory = zeros(7, numIterations + 1);
    jointTrajectory(:, 1) = startJointConfig;
    velocity = zeros(7, 1);
    
    % Initialize performance metrics
    executionTimes = zeros(1, numIterations); % Array to store execution times
    iterationCount = 0; % To count the number of iterations
    
    % Inverse kinematics loop to move the end-effector to the target position
    bestError = inf; % Initialize best error
    noImprovementCount = 0; % Initialize no improvement counter
    
    for n = 1:numIterations
        tic; % Start the timer for performance measurement
        curJointConfig = jointTrajectory(:, n);
        J_temp = geometricJacobian(robot, curJointConfig', endEffector);
        J = J_temp(4:6, :); % Jacobian for Cartesian velocity
        
        Htmp = getTransform(robot, curJointConfig', endEffector); % Forward kinematics
        endEffectorPos = Htmp(1:3, 4); % End-effector position
        error = targetPosition - endEffectorPos; % Positional error
        
        % Calculate error norm in meters
        errorNorm = norm(error);
        
        % Check for early stopping based on error threshold (0.1 mm)
        if errorNorm <= minErrorThreshold
            fprintf('Stopping early at iteration %d with error %f m\n', n, errorNorm);
            jointTrajectory = jointTrajectory(:, 1:n); % Trim unused entries
            break;
        end
        
        % Adjust damping factor dynamically based on the rate of error reduction
        lambda = initialLambda / (1 + 0.5 * errorNorm);
        
        % Levenberg-Marquardt with quasi-Newton approach (BFGS-like update)
        JtJ = J' * J;
        dq = (JtJ + lambda * eye(size(JtJ))) \ (J' * error);
        
        % Apply momentum
        velocity = momentumFactor * velocity + dq;
        
        % Adaptive step size based on error magnitude and improvement
        stepSize = initialStepSize / (1 + 0.1 * errorNorm);
        jointTrajectory(:, n + 1) = curJointConfig + stepSize * velocity; % Update joint configuration
        
        % Early stopping mechanism
        if errorNorm < bestError - minImprovement
            bestError = errorNorm;
            noImprovementCount = 0;
        else
            noImprovementCount = noImprovementCount + 1;
        end
        
        if noImprovementCount > patience
            fprintf('Stopping due to lack of improvement at iteration %d\n', n);
            jointTrajectory = jointTrajectory(:, 1:n); % Trim unused entries
            break;
        end
        
        % Stop the timer and store the elapsed time
        executionTimes(n) = toc;
        iterationCount = n;
    end
    
    % Calculate performance metrics
    totalExecutionTime = sum(executionTimes(1:iterationCount));
    averageTimePerIteration = totalExecutionTime / iterationCount;
    
    % Display performance results
    fprintf('Performance Test Results:\n');
    fprintf('Total Execution Time: %.6f seconds\n', totalExecutionTime);
    fprintf('Average Time Per Iteration: %.6f seconds\n', averageTimePerIteration);
    fprintf('Number of Iterations: %d\n', iterationCount);
    
    % Plot the execution time per iteration
    figure;
    plot(1:iterationCount, executionTimes(1:iterationCount), 'b', 'LineWidth', 2);
    title('Execution Time per Iteration');
    xlabel('Iteration');
    ylabel('Time (seconds)');
    grid on;
    
    % Plot the positional error over iterations
    errors = zeros(1, size(jointTrajectory, 2));
    for n = 1:size(jointTrajectory, 2)
        curJointConfig = jointTrajectory(:, n);
        Htmp = getTransform(robot, curJointConfig', endEffector); % Forward kinematics
        endEffectorPos = Htmp(1:3, 4); % End-effector position
        errors(n) = norm(targetPosition - endEffectorPos); % Error in meters
    end
    
    figure;
    plot(0:(size(errors, 2) - 1), errors, 'r', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Positional Error (m)');
    title('Positional Error Over Iterations');
    legend('Positional Error');
    grid on;

    % Plot the accuracy results (Figure 3)
    finalJointConfig = jointTrajectory(:, end);
    finalTransform = getTransform(robot, finalJointConfig', endEffector);
    finalEndEffectorPos = finalTransform(1:3, 4);
    
    % Convert the final positional error to millimeters
    finalError = norm(targetPosition - finalEndEffectorPos) * 1000;
    
    fprintf('Final End-Effector Position: [%f, %f, %f]\n', finalEndEffectorPos);
    fprintf('Target Position: [%f, %f, %f]\n', targetPosition);
    fprintf('Final Positional Error: %f mm\n', finalError);

    % Save the joint trajectory to a file
    save('jointTrajectory.mat', 'jointTrajectory');
end
