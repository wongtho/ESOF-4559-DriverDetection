function generateAndSaveDriverMasks(labelTable, imageDS)


    %% 1. Create a masks folder
    if ~exist('masks', 'dir')
        mkdir('masks');
    end

    %% 2. Get driver ID
    uniqueDrivers = unique(labelTable.subject);
    [~, baseNames, ext] = cellfun(@fileparts, imageDS.Files, 'UniformOutput', false);
    imgNameDS = strcat(baseNames, ext);

    %% 3. Go through every driver
    for i = 1:numel(uniqueDrivers)
        driverID = uniqueDrivers{i};
        maskFilePath = fullfile('masks', [driverID, '_mask.mat']);

        % 4. Check if the mask already exists
        if exist(maskFilePath, 'file')
            continue;
        end

        fprintf('Processing driver %s\n', driverID);

        % 5. Filter all images for the current driver
        driverMask = strcmp(labelTable.subject, driverID);
        csvNames = labelTable.img(driverMask);
        driverImageMask = ismember(imgNameDS, csvNames);
        filtedDriverDS = subset(imageDS, driverImageMask);

        %% 6. Creating an Analysis Figure
        figure('Name', ['Analysis for Driver ', driverID]);

        % Display sample image
        subplot(2, 2, 1);
        numSamples = min(6, numel(filtedDriverDS.Files));
        randIndex = randperm(numel(filtedDriverDS.Files), numSamples);
        sampleImages = cell(1, numSamples);
        for k = 1:numSamples
            sampleImages{k} = readimage(filtedDriverDS, randIndex(k));
        end
        montage(sampleImages);
        title(['Sample Images for Driver ', driverID]);

        % 7. Calculate pixel standard deviation mask
        numImages = numel(filtedDriverDS.Files);
        tempImg = readimage(filtedDriverDS, 1);
        imgStack = zeros(size(tempImg,1), size(tempImg,2), numImages, 'double');
        
        for j = 1:numImages
            I = readimage(filtedDriverDS, j);
            I = rgb2gray(I);
            imgStack(:, :, j) = im2double(I);
        end
        
        stdMap = std(imgStack, 0, 3);

        % Show Standard Deviation Plot
        subplot(2, 2, 2);
        imshow(stdMap,[]); 
        title('Standard Deviation Map');
        colorbar;

        % Display a histogram of standard deviations
        subplot(2, 2, 3);
        histogram(stdMap);
        title('Std. Dev. Histogram');
        grid on;

        % 8. Thresholding and refining the mask
        threshold = 0.1; 
        binaryMask = stdMap > threshold;
        binaryMask = imfill(binaryMask, 'holes');
        binaryMask = bwareaopen(binaryMask, 100);

        % Show Mask
        subplot(2, 2, 4);
        imshow(binaryMask);
        title('Final Binary Mask');

        % 9. Save the file
        save(maskFilePath, 'binaryMask');
    end
    
end
