function [final_outline, driverMask] = peelWindowFromAllMasks(image, staticMask)
    
    %close all;
    %image = imread("imgs\train\c9\img_42362.jpg");
    %driverMaskFile = fullfile('masks', 'p016_mask.mat');
    % Load the mask variable from the .mat file
    maskData = load(driverMaskFile);
    staticMask = maskData.binaryMask;

    if size(image, 3) == 3
        I_gray = rgb2gray(image);
    else
        I_gray = image;
    end
    
    threshold = 200;
    maskWindow = I_gray >= threshold;
    maskWindow = imfill(maskWindow, 'holes');
    maskWindow = bwareaopen(maskWindow, 8000);
    
    driverMask = staticMask & ~maskWindow;
    
    % 1 Clean up the final driver mask.
    driverMask = imclose(driverMask, strel('disk', 10));
    driverMask = imfill(driverMask, 'holes');
    driverMask = bwareaopen(driverMask, 500);

   
    faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
    faceBBoxes = faceDetector.step(I_gray); 
    
    if ~isempty(faceBBoxes)

        [~, largestFaceIdx] = max(faceBBoxes(:,3) .* faceBBoxes(:,4));
        faceBBox = faceBBoxes(largestFaceIdx, :);
        
        % Create a mask for the face region
        faceMask = false(size(I_gray));
        x = round(faceBBox(1)); y = round(faceBBox(2));
        w = round(faceBBox(3)); h = round(faceBBox(4));
        
        % Ensure the boundaries are within the image area
        [rows, cols] = size(I_gray);
        x_end = min(cols, x + w - 1);
        y_end = min(rows, y + h - 1);
        x = max(1, x); y = max(1, y);

        faceMask(y:y_end, x:x_end) = true;
        
        % Add the face mask to driverMask
        driverMask = driverMask | faceMask;
    end

    % Morphological Optimization: Refine the window mask with a min filter
    maskWindow = ordfilt2(maskWindow, 1, ones(3,3));

    % --- Partitioned Processing ---

    % 1. Driver Area Processing
    driverAreaImg = I_gray;
    driverAreaImg(~driverMask) = 0; % Isolate the driver using the final driver mask
    enhancedDriverArea = adapthisteq(driverAreaImg); % Enhance with histogram equalization

    % 2. Window Area Processing
    % Isolate the window area, ensuring it doesn't overlap with the driver area
    windowOnlyMask = maskWindow & ~driverMask;
    windowAreaImg = I_gray;
    windowAreaImg(~windowOnlyMask) = 0;
    %windowAreaImg(windowOnlyMask) = 0;
    
    % Apply inverse log transform
    tempImg = double(windowAreaImg);
    c = 90; % Constant from experimental script
    inverseLogWindow = exp(tempImg/c) - 1;
    inverseLogWindow = uint8(255 * mat2gray(inverseLogWindow));

    % --- Image Fusion ---
    fusedImg = enhancedDriverArea + inverseLogWindow;

    % --- Multi-stage Sharpening ---
    doubleFused = im2double(fusedImg);
    
    % 1. Laplacian Sharpening
    laplacian = [0 1 0; 1 -4 1; 0 1 0];
    lap_response = imfilter(doubleFused, laplacian, 'replicate');
    sharpened_lap = doubleFused - lap_response;

    % 2. Unsharp Masking
    blurred = imfilter(sharpened_lap, fspecial('gaussian', 15, 3), 'replicate');
    unsharpMask = sharpened_lap - blurred;
    sharpened_final = sharpened_lap + 1.5 * unsharpMask;

    % --- Final Steps from Original Pipeline ---
    % Blur to reduce noise before edge detection
    blurred_final = imgaussfilt(sharpened_final, 2.5);
    
    % Final output is the edge map
    object_outline = edge(blurred_final, 'sobel');
    %final_outline = edge(blurred_final, 'sobel');

    %figure;
    %subplot(2,2,1); imshow(I_gray); title('Gray image');
    %subplot(2,2,2); imshow(driverMask); title('Driver mask');
    %subplot(2,2,3); imshow(enhancedDriverArea); title('Enhanced driver area');
    %subplot(2,2,4); imshow(windowAreaImg); title('Window area');

    %figure;
    %subplot(2,2,1); imshow(inverseLogWindow); title('Inverse log');
    %subplot(2,2,2); imshow(fusedImg); title('Fused image');
    %subplot(2,2,3); imshow(sharpened_lap); title('Laplace sharpened');
    %subplot(2,2,4); imshow(object_outline); title('Final outline');

    % Dilate the outline (sobel edge) with 3x3 structure element
    dilate_outline = ordfilt2(object_outline, 25, ones(5,5));
    %dilate_outline = ordfilt2(object_outline, 81, ones(9,9));
    % And smooth
    dilate_outline = imfilter(im2double(dilate_outline), fspecial('gaussian', 15, 1), 'replicate');
    %dilate_outline = imfilter(im2double(dilate_outline), fspecial('gaussian', 15, 2), 'replicate');

    double_gray = im2double(I_gray);
    % Blur copy of image
    blurred_original = imfilter(double_gray, fspecial('gaussian', 15, 1), 'replicate');
    %blurred_original = imfilter(double_gray, fspecial('gaussian', 15, 3), 'replicate');
    %blurred_original = double_gray;

    % Sharpen copy of image
    blurred_original_unsharp = imfilter(double_gray, fspecial('gaussian', 15, 1.5), 'replicate');
    unsharpMask_original = double_gray - blurred_original_unsharp;
    sharpened_original = double_gray + 1.5 * unsharpMask_original;

    %figure;
    %subplot(2,2,1); imshow(object_outline); title('Final outline');
    %subplot(2,2,2); imshow(dilate_outline); title('Dilated outline');
    %subplot(2,2,3); imshow(blurred_original); title('Blurred original');
    %subplot(2,2,4); imshow(sharpened_original); title('Sharpened original');

    % Set blurred original to 0 of dilated outline
    %blurred_original(dilate_outline) = 0;

    % Set sharpened original to 1 of dilated outline
    %sharpened_original(~dilate_outline) = 0;

    % Blend blurred and sharpened edge
    combined_image = blurred_original .* (1-dilate_outline) + sharpened_original .* (dilate_outline);

    % Apply combined mask
    masked = combined_image;
    masked(~driverMask | windowOnlyMask) = 0;
    
    %figure;
    %subplot(2,2,1); imshow(blurred_original); title('Blurred original');
    %subplot(2,2,2); imshow(sharpened_original); title('Sharpened original');
    %subplot(2,2,3); imshow(combined_image); title('Combined blurred and sharpened');
    %subplot(2,2,4); imshow(masked); title('Masked combined blurred and sharpened');

    % Equalize
    masked_eq = adapthisteq(masked);

    figure;
    %subplot(1,2,1); imshow(masked); title('Masked combined blurred and sharpened');
    subplot(1,2,1); imshow(masked_eq, []); title('Equalized');
    subplot(1,2,2); imhist(masked_eq); title('Histogram distribution');

    % Reduce quantization level
    reduce_quant = uint8(round(masked_eq * 32));

    figure;
    subplot(1,2,1); imshow(masked_eq); title('Full image');
    subplot(1,2,1); imshow(reduce_quant, []); title('Reduced quantization');
    subplot(1,2,2); imhist(reduce_quant); title('Histogram distribution');

    final_outline = reduce_quant;

end