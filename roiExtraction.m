% Extract person ROI using Yolo v2 prediction model
function image_roi = roiExtraction(image, detector)
    %close all;
    % Select a random image from table
    %randIndex = randi(height(labelTable));
    %imgName = labelTable.img{randIndex};

    % Find the index in imageDatastore
    %dsIndex = find(endsWith(imageDS.Files, imgName), 1);
    %if isempty(dsIndex)
    %    error("Image %s not found in imageDatastore.", imgName);
    %end
    
    %fprintf('Selected index: %d\n', dsIndex);
    %img = readimage(imageDS, dsIndex);
    img = image;
    
    % Using deep learning model
    %detector = yolov2ObjectDetector('darknet19-coco');
    %[bboxes, scores, labels] = detect(detector, img);

    %% Enable on GPU
    [bboxes, scores, labels] = detect(detector, img, 'ExecutionEnvironment','gpu');

    % Only keep person detections
    idx = labels == "person";
    bboxes = bboxes(idx,:);
    scores = scores(idx);
    
    % Add bounding box
    %annotated = insertObjectAnnotation(img, 'rectangle', bboxes, scores);
    %imshow(annotated);
    if isempty(bboxes)
        image_roi = img;
        return;
    end
    
    if size(bboxes,1) > 1
        bboxes = bboxes(1,:); % choose 1 bounding box
    end

    image_roi = imcrop(img, bboxes);


end