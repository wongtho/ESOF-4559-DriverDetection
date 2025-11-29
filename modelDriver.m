function modelDriver()

    baselineImgPath = 'imgs/train';
    
    %% 1. Train model on train dataset - unprocessed
    fprintf('Training baseline model...\n');
    trainModel(baselineImgPath, 'baselineModel');
    
    %% 2. Train model on train dataset - processed images
    fprintf('Training processed model...\n');
    trainModel('processed_dataset/train', 'processedModel');

    %% 3. Run validation on baseline
    fprintf('Testing baseline model...\n');
    [baselineAcc, baselinePerClassAcc] = testModel(baselineImgPath, 'baselineModel');
    
    %% 4. Run validation on processed model
    fprintf('Training processed model...\n');
    [processedAcc, processedPerClassAcc] = testModel('processed_dataset/validation', 'processedModel');

    %% 5 Results
    fprintf('Baseline model accuracy: %.2f%%\n', baselineAcc * 100);
    fprintf('Per-class Accuracy:\n');
    for c = 1:10
        fprintf('  %s: %.2f%% (%d/%d)\n', baselinePerClassAcc{c, 1}, ...
            baselinePerClassAcc{c, 2}, baselinePerClassAcc{c, 3}, ...
            baselinePerClassAcc{c, 4});
    end

    fprintf('Processed model accuracy: %.2f%%\n', processedAcc * 100);
    fprintf('Per-class Accuracy:\n');
    for c = 1:10
        fprintf('  %s: %.2f%% (%d/%d)\n', processedPerClassAcc{c, 1}, ...
            processedPerClassAcc{c, 2}, processedPerClassAcc{c, 3}, ...
            processedPerClassAcc{c, 4});
    end
end