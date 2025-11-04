function analyzeDataDistribution(labelTable, imageData) % imageData is needed for file checks

    % Figure 1: Image Count per Class
    figure('Name', 'Image Count per Class');
    histogram(categorical(labelTable.classname));
    title('Image Count per Class');
    xlabel('Class (classname)');
    ylabel('Number of Images');
    grid on;

    % Figure 2: Class Distribution per Driver
    figure('Name', 'Class Distribution per Driver');
    driverClassCounts = groupsummary(labelTable, {'subject', 'classname'}, 'IncludeEmptyGroups', false);
    countMatrix = unstack(driverClassCounts, 'GroupCount', 'classname');
    
    % Fill any remaining NaNs with 0 (for display robustness)
    countData = table2array(countMatrix(:, 2:end));
    countData(isnan(countData)) = 0; 

    subjectCategorical = categorical(countMatrix.subject);
    bar(subjectCategorical, countData, 'stacked');
    title('Class Distribution per Driver');
    xlabel('Driver ID (subject)');
    ylabel('Number of Images');
    grid on;
    xtickangle(45);
    legend(countMatrix.Properties.VariableNames(2:end), 'Location', 'eastoutside');

    fprintf('\n--- Starting Data Validation ---\n');
    
    % 1. Check if all image files listed in labelTable exist and are readable
    fprintf('1. Checking image file existence and readability...\n');
    missingImages = {};
    unreadableImages = {};
    for i = 1:height(labelTable)
        driverID = labelTable.subject{i};
        className = labelTable.classname{i};
        imgName = labelTable.img{i};
        
        % Assuming image path structure: 'imgs/train/classname/imgName'
        imgPath = fullfile(pwd, 'imgs', 'train', className, imgName); 
        
        if ~exist(imgPath, 'file')
            missingImages{end+1} = imgPath;
        else
            try
                imread(imgPath); % Attempt to read to check for corruption
            catch
                unreadableImages{end+1} = imgPath;
            end
        end
    end
    
    if ~isempty(missingImages)
        fprintf('   WARNING: %d images listed in CSV are missing:\n', numel(missingImages));
        for k = 1:min(5, numel(missingImages)) % Show first 5
            fprintf('      - %s\n', missingImages{k});
        end
        if numel(missingImages) > 5, fprintf('      ...and %d more.\n', numel(missingImages)-5); end
    else
        fprintf('   All images listed in CSV exist.\n');
    end

    if ~isempty(unreadableImages)
        fprintf('   WARNING: %d images listed in CSV are unreadable/corrupt:\n', numel(unreadableImages));
        for k = 1:min(5, numel(unreadableImages)) % Show first 5
            fprintf('      - %s\n', unreadableImages{k});
        end
        if numel(unreadableImages) > 5, fprintf('      ...and %d more.\n', numel(unreadableImages)-5); end
    else
        fprintf('   All images listed in CSV are readable.\n');
    end

    % 2. Check for "orphan" images in the train directory (not in CSV)
    fprintf('2. Checking for orphan images in training directory...\n');
    allImageFilesInDir = imageData.Files; % imageData is imageDatastore for train folder
    [~, allNamesInDir, allExtsInDir] = cellfun(@fileparts, allImageFilesInDir, 'UniformOutput', false);
    allFullNamesInDir = strcat(allNamesInDir, allExtsInDir);
    
    allNamesInCSV = labelTable.img;
    
    orphanImages = setdiff(allFullNamesInDir, allNamesInCSV);
    
    if ~isempty(orphanImages)
        fprintf('   WARNING: %d orphan images found in training directory (not in CSV):\n', numel(orphanImages));
        for k = 1:min(5, numel(orphanImages)) % Show first 5
            fprintf('      - %s\n', orphanImages{k});
        end
        if numel(orphanImages) > 5, fprintf('      ...and %d more.\n', numel(orphanImages)-5); end
    else
        fprintf('   No orphan images found in training directory.\n');
    end
    
    fprintf('--- Data Validation Complete ---\n');
end