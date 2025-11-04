function [augimdsTrain, augimdsValidation, augimdsTest] = dataPreparation()
    %% 1. Loading and validating raw data
    [labelTable, imageData, imdsTest] = loadDataset();
    
    %% 2. Data Analysis
    analyzeDataDistribution(labelTable, imageData); % Pass imageData

    %% 3. Generate Base Static Masks (as an enhancement technique)
    generateAndSaveDriverMasks(labelTable, imageData);

    %% 4. Preprocess Data (Splitting and Custom Read Function)
    [augimdsTrain, augimdsValidation, augimdsTest] = preprocessData(imageData, imdsTest, labelTable);


end