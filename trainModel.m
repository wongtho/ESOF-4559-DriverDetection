function trainModel(trainPath, modelName)
    % Check if model is found
    if isfile([modelName, '.mat'])
        fprintf('Model exist: %s\n', modelName);
        return
    end
    %% SET PATHS
    csvFile = 'processed_dataset/train/train_imgs_list.csv'; % Lookup table
    trainFolder = trainPath;

    % CSV structure: subject, classname, img
    T = readtable(csvFile);

    % Convert classname to categorical properly
    T.classname = categorical(cellstr(T.classname));

    numImgs = height(T);

    %% HOG PARAMETERS
    cellSize = [8 8];
    hogImgSize = [160 120];   % resize for consistent vector length
    %% Baseline vs Processed Model
    %  hogSize  |     Model Size    |    Accuracy
    % [160 120] | 640.2MB - 556.4MB | 99.46% - 99.46%
    % [128 96]  | 387.6MB - 348.5MB | 99.41% - 99.45%
    % [80 60]   | 126.4MB - 121.6MB | 99.00% - 99.06%

    % Replace with .jpg if in unprocessed folder
    if strcmp(char(trainFolder), 'imgs/train')
        firstImgName = strrep(T.img{1}, '.png', '.jpg');
    else
        firstImgName = T.img{1}; % Use default extension
    end
    % Load first image to get feature length
    firstImg = imread(fullfile(trainFolder, char(T.classname(1)), firstImgName));
    if size(firstImg,3)==3, firstImg = rgb2gray(firstImg); end
    firstImg = imresize(firstImg, hogImgSize);
    featLength = length(extractHOGFeatures(firstImg, "CellSize", cellSize));
    %% Preallocate
    hogFeatures = zeros(numImgs, featLength);
    hogLabels = categorical(strings(numImgs,1));

    fprintf("Extracting HOG features...\n");

    %% Loop through train images
    for i = 1:numImgs
        className = T.classname(i); % ex "c0", "c1", ... "c9"
        imgName   = T.img{i}; % ex "img_42362.png"

        % Replace with .jpg if in unprocessed folder
        if strcmp(char(trainFolder), 'imgs/train')
            imgName = strrep(imgName, '.png', '.jpg');
        end
        
        % "processed_dataset/train/c9/img_42362.png
        fullPath = fullfile(trainFolder, char(className), imgName);

        if ~isfile(fullPath)
            fprintf("Missing: %s\n", fullPath);
            continue;
        end

        % Load, HOG
        I = imread(fullPath);
        if size(I,3)==3, I = rgb2gray(I); end
        I = imresize(I, hogImgSize);

        feat = extractHOGFeatures(I, "CellSize", cellSize);

        hogFeatures(i,:) = feat;
        hogLabels(i) = className;

        if mod(i, 100) == 0
            fprintf('  HOG feature extracted %d / %d images.\n', i, numImgs);
        end
    end

    fprintf("HOG extraction finished.\n");


    %% TRAIN CLASSIFIER (ECOC with SVM learners)
    fprintf("Training SVM model...\n");

    template = templateSVM('KernelFunction', 'linear');

    model = fitcecoc(hogFeatures, hogLabels, 'Learners', template);

    fprintf("Training completed!\n");


    %% SAVE MODEL
    save([modelName, '.mat'], 'model', 'cellSize', 'hogImgSize');
    fprintf("Model saved as %s.mat\n", modelName);
end