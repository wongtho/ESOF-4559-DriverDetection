function [final_outline, driverMask] = peelWindowFromAllMasks(image, staticMask)


    if size(image, 3) == 3
        I_gray = rgb2gray(image);
    else
        I_gray = image;
    end
    
    threshold = 180;
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

    % Enhancement and Feature Extraction
    isolatedDriverGray = I_gray;
    isolatedDriverGray(~driverMask) = 0;
    
    enhancedDriver = adapthisteq(isolatedDriverGray);
    sharpenedDriver = imsharpen(enhancedDriver, 'Radius', 2, 'Amount', 1.5);
    blurredDriver = imgaussfilt(sharpenedDriver, 1.5);
    
    final_outline = edge(blurredDriver, 'sobel');
end