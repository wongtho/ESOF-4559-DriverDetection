function dataPreparation()
    %% 1. Loading and validating data
    [labelTable, imageData, imdsTest] = loadDataset();
    
    %% 2. Data Analysis
    % This function is used to visualize data distribution
    analyzeDataDistribution(labelTable);
    
    %% 3. Data preprocessing
    [trainData, valData, testData] = preprocessData(imageData, imdsTest);

        % Print information

        fprintf('Training set contains %d samples.\n', numel(trainData.Files));

        fprintf('Validation set contains %d samples.\n', numel(valData.Files));

        fprintf('Test set contains %d samples.\n', numel(testData.Files));
end
