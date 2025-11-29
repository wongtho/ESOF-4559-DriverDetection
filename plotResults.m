% Baseline accuracies
baseline = [45.68 70.11 33.07 70.13 46.86 74.87 57.20 70.60 41.50 23.47];

% Processed accuracies
processed = [61.49 52.51 46.35 57.79 72.91 73.67 58.63 62.17 46.40 38.61];

% Compute change
delta = processed - baseline;

% Actual behavior labels (StateFarm dataset)
behaviors = { ...
    'safe driving', ...
    'texting right', ...
    'talking right', ...
    'texting left', ...
    'talking left', ...
    'operating radio', ...
    'drinking', ...
    'reaching behind', ...
    'hair/makeup', ...
    'talking to passenger' ...
    };

figure;
bar(delta);
set(gca, 'XTick', 1:10, 'XTickLabel', behaviors);
xtickangle(45); % tilt labels for readability

ylabel('Accuracy Change (%)');
title('Accuracy Improvement per Behavior (Processed - Baseline)');

% Label each bar with numeric value
ytips = delta;
xtips = 1:10;
labels = string(round(delta,2));

text(xtips, ytips, labels, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', ...
    'FontSize', 11);

grid on;

% Behavior labels
behaviors = {
    'safe driving'
    'texting right'
    'talking right'
    'texting left'
    'talking left'
    'operating radio'
    'drinking'
    'reaching behind'
    'hair/makeup'
    'talking to passenger'
};

% Baseline per-class accuracy (%)
baselineAcc = [45.68; 70.11; 33.07; 70.13; 46.86; 74.87; 57.20; 70.60; 41.50; 23.47];

% Baseline correct and total
baselineCorrect = [344; 502; 254; 540; 358; 563; 441; 461; 271; 169];
baselineTotal   = [753; 716; 768; 770; 764; 752; 771; 653; 653; 720];

% Processed per-class accuracy (%)
processedAcc = [61.49; 52.51; 46.35; 57.79; 72.91; 73.67; 58.63; 62.17; 46.40; 38.61];

% Processed correct and total
processedCorrect = [463; 376; 356; 445; 557; 554; 452; 406; 303; 278];
processedTotal   = [753; 716; 768; 770; 764; 752; 771; 653; 653; 720];

% Create MATLAB table
T = table(behaviors, baselineAcc, processedAcc, ...
          baselineCorrect, baselineTotal, ...
          processedCorrect, processedTotal);

% Display table
disp(T);