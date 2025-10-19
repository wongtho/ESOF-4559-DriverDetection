function analyzeDataDistribution(labelTable)

    % Draw a bar chart to show the number of samples in each category
    figure;
    histogram(categorical(labelTable.classname));
    
    title('Image Count per Class');
    xlabel('Class (classname)');
    ylabel('Number of Images');
    grid on;

end