function testIKIntegration()
    % Load the robot model
    robot = loadrobot('kinovaGen3', 'DataFormat', 'row', 'Gravity', [0 0 -9.81]);
    endEffector = "EndEffector_Link";

    % Initial configuration of the robot (as a row vector)
    startJointConfig = [0 0.1 0.0 0.2 -0.0 1.0 0.0]';
    
    % Define a series of 3D target positions for the end-effector
    targetPositions = [
        0.7, 0.1, 0.3;  % Target 1
        0.6, 0.15, 0.35; % Target 2
        0.65, 0.05, 0.25 % Target 3
    ];

    % Set IK and solver parameters
    numIterations = 2000; % Maximum number of iterations
    initialStepSize = 0.01; % Initial step size
    initialLambda = 0.1; % Initial Levenberg-Marquardt damping factor
    minErrorThreshold = 0.0001; % Error threshold for stopping (0.1 mm)
    momentumFactor = 0.9; % Momentum factor
    patience = 100; % Early stopping patience
    minImprovement = 1e-6; % Minimum improvement to reset patience

    % Preallocate arrays for storing results
    jointTrajectory = zeros(7, numIterations + 1);
    velocity = zeros(7, 1);
    
    % Iterate over each target position
    for i = 1:size(targetPositions, 1)
        targetPosition = targetPositions(i, :)';
        jointTrajectory(:, 1) = startJointConfig; % Reset the trajectory for each target
        bestError = inf; % Reset best error
        noImprovementCount = 0; % Reset improvement counter
        
        % Inverse kinematics loop to move the end-effector to the target position
        for n = 1:numIterations
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
                fprintf('Stopping early at iteration %d with error %f m for target %d\n', n, errorNorm, i);
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
                fprintf('Stopping due to lack of improvement at iteration %d for target %d\n', n, i);
                jointTrajectory = jointTrajectory(:, 1:n); % Trim unused entries
                break;
            end
        end
        
        % Final configuration and error calculation
        finalJointConfig = jointTrajectory(:, end);
        finalTransform = getTransform(robot, finalJointConfig', endEffector);
        finalEndEffectorPos = finalTransform(1:3, 4);
        finalError = norm(targetPosition - finalEndEffectorPos) * 1000; % Error in mm
        
        % Visualization of the robot's position after reaching each target
        show(robot, finalJointConfig', 'PreservePlot', false);
        hold on;
        plot3(targetPosition(1), targetPosition(2), targetPosition(3), 'ro', 'MarkerSize', 10);
        drawnow;

        % Display and assert accuracy
        fprintf('Target %d Final End-Effector Position: [%f, %f, %f]\n', i, finalEndEffectorPos);
        fprintf('Target Position: [%f, %f, %f]\n', targetPosition);
        fprintf('Final Positional Error: %f mm\n', finalError);
        
        assert(finalError < 0.1, sprintf('Target %d not reached accurately. Error: %.4f mm', i, finalError));
        
        % Update the initial configuration for the next target
        startJointConfig = finalJointConfig;
    end
    
    disp('All integration tests passed successfully.');
end

