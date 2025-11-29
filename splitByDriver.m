function [imdsTrain, labelTrainTable, imdsValidation, labelValidationTable] = splitByDriver(labelTable)
    
    T = labelTable;
    imgFolder = 'imgs/train';
    fullPaths = fullfile(imgFolder, labelTable.classname, labelTable.img);

    imds = imageDatastore(fullPaths, 'Labels', categorical(T.classname));
    %numel(imds.Files)
    %height(T)
    
    uniqueDrivers = unique(T.subject);

    % Shuffle drivers
    rng(123); % Reproducability
    uniqueDrivers = uniqueDrivers(randperm(numel(uniqueDrivers)));
    
    % 70% of drivers for training
    numTrainDrivers = round(0.7 * numel(uniqueDrivers));
    trainDrivers = uniqueDrivers(1:numTrainDrivers);

    trainMask = ismember(T.subject, trainDrivers);
    validMask = ~trainMask;

    % Create separate tables
    labelTrainTable = T(trainMask, :);
    labelValidationTable  = T(validMask, :);
    
    % Separate train and validations by driver
    imdsTrain = subset(imds, find(trainMask));
    imdsValidation = subset(imds, find(validMask));

    %disp("Training Drivers:");
    %disp(trainDrivers);
    
    %disp("Test Drivers:");
    %disp(setdiff(uniqueDrivers, trainDrivers));

    disp("Training Table:");
    disp(head(labelTrainTable));

    disp("Validation Table:");
    disp(head(labelValidationTable));
    
    disp("Train Table:");
    disp(height(labelTrainTable));

    disp("Validation Table:");
    disp(height(labelValidationTable));
end