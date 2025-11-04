function [augimdsTrain, augimdsValidation, augimdsTest] = preprocessData(imageData, imdsTest, labelTable)
    %% 1. Define model input image size
    inputSize = [480 640 1]; 

    %% 2. Split raw data into training and validation sets
    [imdsTrain, imdsValidation] = splitEachLabel(imageData, 0.7, 'randomized');

    % Shuffle the training data to ensure batches contain a mix of classes.
    imdsTrain = shuffle(imdsTrain);

    fprintf('Raw data split into %d training images and %d validation images.\n', numel(imdsTrain.Files), numel(imdsValidation.Files));

    %% 3. Define Custom Read Function
    
    % Nested function to access labelTable and other variables
    function processedImage = customReadFunction(imgPath)
        % Load original image
        I_orig = imread(imgPath);
        
        % Determine driverID from image path
        [~, imgName, ext] = fileparts(imgPath);
        imgFullName = [imgName, ext];
        imgEntry = strcmp(labelTable.img, imgFullName);
        driverID = labelTable.subject{imgEntry};
        
        % Load the driver's static mask
        maskFilePath = fullfile('masks', [driverID, '_mask.mat']);
        maskData = load(maskFilePath);
        staticMask = maskData.binaryMask;

        final_outline = peelWindowFromAllMasks(I_orig, staticMask);
        
        % Resize the output to the model's input size
        processedImage = imresize(final_outline, inputSize(1:2));
        processedImage = single(processedImage);
    end

    %% 4. Attach Custom Read Function to imageDatastore's ReadFcn property
    imdsTrain.ReadFcn = @customReadFunction;
    imdsValidation.ReadFcn = @customReadFunction;
    imdsTest.ReadFcn = @customReadFunction;

    %% 5. Create Augmented Image Datastores
    % Now, augmentedImageDatastore will automatically use the ReadFcn of the underlying imageDatastore.
    
    augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, imdsTrain.Labels);

    augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, imdsValidation.Labels);

    augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, imdsTest.Labels);

    fprintf('Augmented datastores created.\n');
end