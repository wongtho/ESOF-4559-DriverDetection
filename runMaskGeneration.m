function runMaskGeneration()
    
    %% 1. Generation phase
    [labelTable, imageData, imdsTest] = loadDataset();
    generateAndSaveDriverMasks(labelTable, imageData);

    %% 2. Display phase
    displayAllMasks(labelTable);

end
