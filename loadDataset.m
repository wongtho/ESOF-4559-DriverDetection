function [labelTable, imageData, imdsTest] = loadDataset()


    %% 1. Loading a CSV file
    labelTable = loadClassificationTable('driver_imgs_list.csv');

    %% 2. Loading training image data
    trainFolder = fullfile(pwd, 'imgs', 'train');
    imageData = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    fprintf('Loaded %d raw training images from %s.\n', numel(imageData.Files), trainFolder);

    %% 3. Loading test image data
    testFolder = fullfile(pwd, 'imgs', 'test');
    imdsTest = imageDatastore(testFolder, 'IncludeSubfolders', true);
    fprintf('Loaded %d raw test images from %s.\n', numel(imdsTest.Files), testFolder);

end


%% Load csv file containing image, classification, driverID
function classificationTable = loadClassificationTable(pathToCSV) 
    classificationTable = readtable(pathToCSV);
    % Display number of lines in table
    numLines = height(classificationTable);
    fprintf('Loaded %d rows of labels from %s.\n', numLines, pathToCSV);
end