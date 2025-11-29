function dataPreparation()
    % This function is the all-in-one preprocessing and saving script.
    % It processes all raw images and saves the results to the 'processed_dataset'
    % directory, to be used by the training script.

    fprintf('--- Running Data Preparation ---\n');
    fprintf('This is a one-time setup that will take a long time.\n');
    
    outputDir = 'processed_dataset';
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    %% 1. Load Initial Data
    fprintf('Loading dataset...\n');
    [labelTable, imageData, ~] = loadDataset();

    %% 2. Analyze Data and Generate Masks
    %fprintf('Analyzing data distribution...\n');
    %analyzeDataDistribution(labelTable, imageData);

    %% 3. Split Data into Training and Validation sets
    fprintf('Splitting data into training and validation sets...\n');
    %[imdsTrain, imdsValidation] = splitEachLabel(imageData, 0.7, 'randomized');
    [imdsTrain, trainTable, imdsValidation, validationTable] = splitByDriver(labelTable);

    %fprintf('Generate general driver mask based on train set...\n');
    %generateAndSaveDriverMasks(labelTable, imageData);
    %generateAndSaveDriverMasks(labelTable, imdsTrain);

    %% 4. Process and Save the Datastores
    fprintf('\nProcessing and saving TRAINING set...\n');
    processAndSave(imdsTrain, fullfile(outputDir, 'train'), trainTable, 'train_imgs_list');

    fprintf('\nProcessing and saving VALIDATION set...\n');
    processAndSave(imdsValidation, fullfile(outputDir, 'validation'), validationTable, 'validation_imgs_list');

    fprintf('\n--- Data Preparation Complete ---\n');
    fprintf('Preprocessed data has been saved to the "%s" directory.\n', outputDir);

    % --- Nested Helper Function ---
    function processAndSave(imds, destinationFolder, labelTable, csvFileName)
        % Create YOLO detector once (GPU)
        detector = yolov2ObjectDetector('darknet19-coco');

        % Helper function to read, process, and save a datastore.
        numFiles = numel(imds.Files);
        % Initialize a cell array to store label-filename pairs
        savedData = cell(numFiles, 2); % columns: classname, img
        for i = 1:numFiles
            % Get file path and label
            imgPath = imds.Files{i};
            label = imds.Labels(i);
            
            % Read the original image
            I_orig = imread(imgPath);

            % Create the output directory and save the image
            outputLabelDir = fullfile(destinationFolder, char(label));
            if ~exist(outputLabelDir, 'dir')
                mkdir(outputLabelDir);
            end
            
            [~, originalFilename, ~] = fileparts(imgPath);
            outputPath = fullfile(outputLabelDir, [originalFilename, '.png']);
            

            % Store filename and label info
            savedData{i, 1} = char(label);
            savedData{i, 2} = [originalFilename, '.png'];

            % Check if the file already exists to avoid overwriting
            if exist(outputPath, 'file')
                %fprintf('File already exists: %s. Skipping...\n', outputPath);
                continue;
            end

            % Determine driverID from image path to get the correct mask
            %[~, imgName, ext] = fileparts(imgPath);
            %imgFullName = [imgName, ext];
            %imgEntry = strcmp(labelTable.img, imgFullName);
            %driverID = labelTable.subject{imgEntry};
            
            % Load the driver's static mask
            %maskFilePath = fullfile('masks', [driverID, '_mask.mat']);
            %maskData = load(maskFilePath);
            %staticMask = maskData.binaryMask;
    
            % Run the full preprocessing pipeline
            %[final_outline, ~] = peelWindowFromAllMasks(I_orig, staticMask);

            % Find ROI region
            croppedImg = roiExtraction(I_orig, detector);
        
            % Obtain image edge + apply window mask
            processedImg = featureEnhance(croppedImg);
            
            % Write image file
            imwrite(processedImg, outputPath);
            
            % Print progress
            if mod(i, 100) == 0
                fprintf('  Processed and saved %d / %d images.\n', i, numFiles);
            end
        end
        % Write to CSV with class and filename columns
        T = cell2table(savedData, 'VariableNames', {'classname', 'img'});
        csvFilePath = fullfile(destinationFolder, [csvFileName, '.csv']);
        writetable(T, csvFilePath);
        fprintf('  Finished saving %d images to %s.\n', numFiles, destinationFolder);
    end
end
