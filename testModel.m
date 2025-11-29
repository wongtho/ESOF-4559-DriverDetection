function [accuracy, perClassAcc] = testModel(validationPath, modelName)
    %% Load model
    svmModel = load([modelName, '.mat']);
    %% Set path
    csvFile = 'processed_dataset/validation/validation_imgs_list.csv'; % Lookup table
    validationFolder = validationPath;

    % CSV structure: classname, img
    T = readtable(csvFile);

    % Convert classname to categorical properly
    T.classname = categorical(cellstr(T.classname));

    classes = categories(T.classname);
    numClasses = numel(classes);

    totalImages = height(T);
    correctPredictions = 0;

    % Initialize per-class counters
    classCounts = zeros(numClasses,1);
    classCorrect = zeros(numClasses,1);

    % Set timer
    tic;

    for i = 1:totalImages
        trueLabel = T.classname(i);
        imgName = T.img{i};
        % Replace with .jpg if in unprocessed folder
        if strcmp(char(validationPath), 'imgs/train')
            imgName = strrep(imgName, '.png', '.jpg');
        end

        imgPath = fullfile(validationFolder, char(trueLabel), imgName);

        I = imread(imgPath);
        I = imresize(I, svmModel.hogImgSize);
        hogFeatures = extractHOGFeatures(I, 'CellSize', svmModel.cellSize);

        [predictedLabel, ~] = predict(svmModel.model, hogFeatures);

         % Update total and per-class counters
        classIdx = find(strcmp(classes, char(trueLabel)));
        classCounts(classIdx) = classCounts(classIdx) + 1;

        if strcmp(char(predictedLabel), char(trueLabel))
            correctPredictions = correctPredictions + 1;
            classCorrect(classIdx) = classCorrect(classIdx) + 1;
        end

        if mod(i, 100) == 0
            fprintf('  Predicted %d / %d images.\n', i, totalImages);
        end
    end
    
    % Elapsed time
    elapsedTime = toc;
    fprintf('Elapsed time: %.2f seconds\n', elapsedTime);
    
    % Overall accuracy
    accuracy = correctPredictions / totalImages;
    %fprintf('Validation Accuracy Rate: %.2f%%\n', accuracy * 100);

    % Per-class accuracy
    perClassAcc = cell(numClasses, 4); % preallocate
    % Per-class accuracy
    %fprintf('Per-class Accuracy:\n');
    for c = 1:numClasses
        classAcc = classCorrect(c) / classCounts(c);
        perClassAcc{c, 1} = classes{c}; % class name
        perClassAcc{c, 2} = classAcc*100; % accuracy
        perClassAcc{c, 3} = classCorrect(c); % correct
        perClassAcc{c, 4} = classCounts(c); % counts
        %fprintf('  %s: %.2f%% (%d/%d)\n', classes{c}, classAcc*100, classCorrect(c), classCounts(c));
    end
    
end
