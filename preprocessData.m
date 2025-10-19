function [augimdsTrain, augimdsValidation, augimdsTest] = preprocessData(imageData, imdsTest)

    %% 1. Define the input image size
    inputSize = [480 640 3];

    %% 2. Split the data into training and validation sets (80/20 split)
    [imdsTrain, imdsValidation] = splitEachLabel(imageData, 0.8, 'randomized');

    % Shuffle the training data to ensure batches contain a mix of classes.
    imdsTrain = shuffle(imdsTrain);

    fprintf('Split data into %d training images and %d validation images.\n', numel(imdsTrain.Files), numel(imdsValidation.Files));

    %% 3. Defining Data Augmenters

    %% 4. Create a data store
    augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'ColorPreprocessing', 'gray2rgb');
    augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, 'ColorPreprocessing', 'gray2rgb');
    augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, 'ColorPreprocessing', 'gray2rgb');

    %% 5. Display a batch of processed images for verification
    % Display a batch of processed images with their labels for verification.
    figure;
    sampleBatch = preview(augimdsTrain);
    for i = 1:6
        subplot(2,3,i);
        img = sampleBatch{i,1}{1};
        imshow(img);  
        label = sampleBatch{i,2};
         title(sprintf('Label: %s', string(label)));
    end

end