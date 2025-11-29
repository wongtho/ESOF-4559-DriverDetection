% Enhance image by applying overall/contrast enhancement
% Blend outline sharpening and hand+texting enhancement with overall
function processedImg = featureEnhance(roiImg)
    % Convert input to grayscale
    grayImg = rgb2gray(roiImg); % Cropped image

    % Remove window
    threshold = 200;
    maskWindow = grayImg >= threshold;
    maskWindow = imfill(maskWindow, 'holes');
    maskWindow = bwareaopen(maskWindow, 8000);

    % Morphological Optimization: Refine the window mask with a min filter
    maskWindow = ordfilt2(maskWindow, 1, ones(3,3));

    % Apply window mask
    windowMasked = grayImg;
    windowMasked(maskWindow) = 0;

    % Equalize
    equalizedImg = adapthisteq(windowMasked);

    % Apply inverse log transform outside of mask
    tempImg = double(equalizedImg);
    c = 90; % Constant from experimental script
    inverseLogWindow = exp(tempImg/c) - 1;
    inverseLogWindow = uint8(255 * mat2gray(inverseLogWindow));

    % Convert to double for operation
    doubleImg = im2double(inverseLogWindow);

    % 1. Laplacian Sharpening
    laplacian = [0 1 0; 1 -4 1; 0 1 0];
    lap_response = imfilter(doubleImg, laplacian, 'replicate');
    sharpened_lap = doubleImg - lap_response;

    % 2. Unsharp Masking
    blurred = imfilter(sharpened_lap, fspecial('gaussian', 15, 2), 'replicate');
    unsharpMask = sharpened_lap - blurred;
    sharpened_final = sharpened_lap + 1.5 * unsharpMask;

    blurred_final = imgaussfilt(sharpened_final, 2.5);
    
    % Final output is the edge map
    object_outline = edge(blurred_final, 'sobel');

    % SE
    se_disk = strel('disk', 3);

    % Dilate outline
    dilated_outline = imdilate(object_outline, se_disk);

    % Smooted dilated_disk
    smoothedDilatedDisk = imfilter(dilated_outline, fspecial('gaussian', 15, 1), 'replicate');

    % 1. Unsharp Masking
    blurred = imfilter(grayImg, fspecial('gaussian', 15, 1), 'replicate');
    unsharpMask = grayImg - blurred;
    sharpened_final = grayImg + 3.5 * unsharpMask;

    % Combine sharpened with edge map
    I = im2double(sharpened_final);
    E = imgaussfilt(double(object_outline) / 255, 1);
    E = mat2gray(E);

    alpha = 0.45;
    enhanced = I + alpha * E .* (I - im2double(grayImg));

    eq_img = adapthisteq(enhanced, 'NumTiles', [16 16], 'ClipLimit',0.001);

    % Apply original to enhanced based on mask
    combined_image = I .* (1-smoothedDilatedDisk) + im2double(eq_img) .* (smoothedDilatedDisk);

    %processedImg = im2uint8(eq_img);
    %processedImg = combined_image;

    %figure;
    %subplot(2,2,1); imshow(sharpened_final); title('Original');
    %subplot(2,2,2); imshow(processedImg); title('Combined');
    %subplot(2,2,3); imshow(grayImg); title('Default');
    %subplot(2,2,4); imshow(smoothedDilatedDisk); title('Outline');
    %subplot(2,2,3); imshow(sharpened_final); title('Unsharped');
    %subplot(2,2,4); imshow(processedImg); title('processed');

    %% Low accuracy on text
    [height, width, ~] = size(grayImg);

    % Calculate center quarter region
    mask = zeros(height, width);  % initialize mask
    h_start = floor(1*height/5) + 1;
    h_end   = floor(4*height/5);
    w_start = floor(2*width/5) + 1;
    w_end   = floor(4*width/5);
    
    % Set center region to 1
    mask(h_start:h_end, w_start:w_end) = 1;
    mask_text = imfilter(mask, fspecial('gaussian', 25, 3), 'replicate');
    
    masked_text = combined_image .* (1-mask_text) + im2double(grayImg) .* (mask_text);

    %figure;
    %subplot(1,2,1); imshow(combined_image);
    %subplot(1,2,2); imshow(masked_text);
    
    processedImg = masked_text;
end