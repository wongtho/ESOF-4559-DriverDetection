function displayAllMasks(labelTable)
    
    uniqueDrivers = unique(labelTable.subject);
    numDrivers = numel(uniqueDrivers);
    
    % Create a window to display all subgraphs
    figure('Name', 'All Generated Driver Masks');
    
    gridRows = floor(sqrt(numDrivers));
    gridCols = ceil(numDrivers / gridRows);
    
    for i = 1:numDrivers
        driverID = uniqueDrivers{i};
        maskFilePath = fullfile('masks', [driverID, '_mask.mat']);
        
        subplot(gridRows, gridCols, i);
        maskData = load(maskFilePath);
        imshow(maskData.binaryMask);
        title(driverID, 'Color', 'k'); 
    end
    
end
