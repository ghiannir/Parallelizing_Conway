
program_type = {'cuda', 'openmp', 'sequential'};


for index = 1:length(program_type)
    
    file_name = strcat("stats_", program_type{index}, "_cleaned.csv");

    % Load data from the CSV file
    % Replace 'data.csv' with the path to your CSV file
    data = readmatrix(file_name, 'NumHeaderLines', 1);
    
    % Extract x, y, and z columns
    x = data(:, 1);  % First column as x
    y = data(:, 2);  % Second column as y
    z = data(:, 3);  % Third column as z
    
    % Create a grid for x and y
    % Assuming x and y have unique values to form a grid
    [X_unique, Y_unique] = meshgrid(unique(x), unique(y));
    
    % Reshape z to match the size of the meshgrid
    Z = griddata(x, y, z, X_unique, Y_unique);
    

    figure;
    % Create surface plot
    surf(X_unique, Y_unique, Z);
    
    % Add labels and title
    xlabel('Iterations');
    ylabel('Dimensions');
    zlabel('Time - [ms]');
    title(program_type{index});
    
    % Optional: Improve the visualization
    shading flat;  % Smooth shading
    colorbar;        % Display color bar

end



