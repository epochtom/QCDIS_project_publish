%% PCA + KNN Classifier + Performance Plot
% NO TOOLBOXES NEEDED (uses only base MATLAB)
% Works in R2025b with base MATLAB only

clear; clc; close all;

%% ============================= CONFIG =============================
CFG.DATASET_PATH   = '/app/upload/dataset.csv';  % Relative path (same folder)
CFG.TEST_SIZE      = 0.2;
CFG.RANDOM_STATE   = 42;
CFG.N_COMPONENTS   = 8;
CFG.K_NEIGHBORS    = 5;  % Number of neighbors for KNN
CFG.path_saving_plot = "/app/output";
CFG.performance_plot_name = 'performance_summary.png';
CFG.PLOT_SAVE_PATH = fullfile(CFG.path_saving_plot, CFG.performance_plot_name);

rng(CFG.RANDOM_STATE);

%% ============================= CENTERING UTILITY =============================
centerString = @(str, width) centerStringImpl(str, width);

%% ============================= MAIN =============================
fprintf('%s\n', repmat('=', 1, 70));
fprintf('%s\n', centerString(' PCA + KNN CLASSIFIER ', 70));
fprintf('%s\n', repmat('=', 1, 70));

% --- Load Data ---
try
    data = readtable(CFG.DATASET_PATH, 'VariableNamingRule', 'preserve');
    fprintf('Loaded -> %d samples, %d features\n', height(data), width(data));
catch ME
    error('Failed to load dataset: %s\nMake sure dataset.csv is in the current folder.', ME.message);
end

% --- Find Target Column ---
targetCol = findTargetColumn(data);
fprintf('Target -> %s | Classes = %d\n', targetCol, numel(unique(data.(targetCol))));

y = data.(targetCol);
X = removevars(data, targetCol);

% --- Encode Labels ---
[~, ~, y_encoded] = unique(y);
y_encoded = y_encoded - 1;  % 0-indexed

% --- Manual Stratified Train/Test Split ---
[y_grp, ~, grp_idx] = unique(y_encoded);
n_per_group = histcounts(y_encoded, 0.5:1:numel(y_grp)+0.5);
test_idx = [];

for g = 1:numel(y_grp)
    idx_g = find(y_encoded == y_grp(g));
    n_test_g = round(CFG.TEST_SIZE * n_per_group(g));
    perm = randperm(numel(idx_g));
    test_idx = [test_idx; idx_g(perm(1:n_test_g))];
end

train_idx = setdiff(1:height(data), test_idx);

X_tr = X(train_idx, :);
X_te = X(test_idx, :);
y_tr = y_encoded(train_idx);
y_te = y_encoded(test_idx);

fprintf('Train: %d samples | Test: %d samples\n', numel(train_idx), numel(test_idx));

% --- Standardize Features ---
mu = mean(X_tr{:,:});
sigma = std(X_tr{:,:});
sigma(sigma == 0) = 1;

X_tr_s = (X_tr{:,:} - mu) ./ sigma;
X_te_s = (X_te{:,:} - mu) ./ sigma;

% --- Classical PCA (Manual SVD) ---
fprintf('  -> Applying classical PCA (n_components=%d)...\n', CFG.N_COMPONENTS);
[U, S, V] = svd(X_tr_s, 'econ');
coeff = V(:, 1:CFG.N_COMPONENTS);
X_tr_pca = X_tr_s * coeff;
X_te_pca = X_te_s * coeff;

eigenvalues = diag(S).^2 / (size(X_tr_s, 1) - 1);
explained = sum(eigenvalues(1:CFG.N_COMPONENTS)) / sum(eigenvalues) * 100;
fprintf('Reduced -> [%d, %d] | Explained: %.2f%%\n', size(X_tr_pca), explained);

% --- Handle Single-Class Case ---
if numel(unique(y_tr)) < 2
    fprintf('Only 1 class -> dummy model\n');
    pred = zeros(size(y_te));
    acc = 0.0;
    f1 = 0.0;
else
    % --- KNN Classifier (Pure MATLAB, no toolboxes) ---
    fprintf('  -> Training KNN classifier (k=%d)...\n', CFG.K_NEIGHBORS);
    pred = simpleKNN(X_tr_pca, y_tr, X_te_pca, CFG.K_NEIGHBORS);

    % --- Metrics ---
    acc = mean(pred == y_te);
    classes = unique([y_tr; y_te]);
    f1_per_class = zeros(numel(classes), 1);
    weights = zeros(numel(classes), 1);
    for i = 1:numel(classes)
        c = classes(i);
        tp = sum(pred == c & y_te == c);
        fp = sum(pred == c & y_te ~= c);
        fn = sum(pred ~= c & y_te == c);
        precision = tp / (tp + fp + eps);
        recall = tp / (tp + fn + eps);
        f1_per_class(i) = 2 * precision * recall / (precision + recall + eps);
        weights(i) = sum(y_te == c);
    end
    f1 = sum(f1_per_class .* weights) / sum(weights);
end

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf(' SUCCESS! Accuracy: %.6f | Weighted F1: %.6f\n', acc, f1);
fprintf(' MODEL: PCA(%d) -> KNN (k=%d)\n', CFG.N_COMPONENTS, CFG.K_NEIGHBORS);
fprintf('%s\n', repmat('=', 1, 70));

% --- Save Performance Plot ---
plotOverallPerformance(acc, f1, CFG.PLOT_SAVE_PATH);

%% ============================= FUNCTIONS =============================

function col = findTargetColumn(tbl)
    if any(strcmp(tbl.Properties.VariableNames, 'label'))
        col = 'label';
    elseif height(tbl) > 0 && numel(unique(tbl{:,1})) <= 20 && isnumeric(tbl{:,1})
        col = tbl.Properties.VariableNames{1};
    else
        col = tbl.Properties.VariableNames{end};
    end
end

function plotOverallPerformance(acc, f1, savePath)
    [saveDir, ~, ~] = fileparts(savePath);
    if ~isempty(saveDir) && ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    figure('Position', [100, 100, 600, 500]);
    metrics = {'Accuracy', 'Weighted F1'};
    values = [acc, f1];
    colors = [0.3, 0.7, 0.3; 0.13, 0.59, 0.95];

    barHandle = bar(1:2, values, 'FaceColor', 'flat');
    barHandle.CData = colors;
    set(barHandle, 'BarWidth', 0.6, 'EdgeColor', 'k', 'LineWidth', 1.2);

    ylim([0, 1.05]);
    ylabel('Score', 'FontSize', 12);
    title('Model Performance (KNN)', 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'XTickLabel', metrics, 'FontSize', 11);

    for i = 1:2
        text(i, values(i) + 0.02, sprintf('%.4f', values(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
    end

    grid on; grid minor; set(gca, 'GridAlpha', 0.3);
    box on;

    print(gcf, savePath, '-dpng', '-r300');
    close(gcf);
    fprintf('Plot saved: %s\n', savePath);
end

function out = centerStringImpl(str, width)
    str = convertCharsToStrings(str);
    n = strlength(str);
    if n >= width
        out = str(1:width);
        return;
    end
    left = floor((width - n) / 2);
    right = width - n - left;
    out = [repmat(' ', 1, left), str, repmat(' ', 1, right)];
end

function pred = simpleKNN(X_train, y_train, X_test, k)
    % Simple K-Nearest Neighbors classifier (pure MATLAB, no toolboxes)
    % X_train: training features [n_train x n_features]
    % y_train: training labels [n_train x 1]
    % X_test: test features [n_test x n_features]
    % k: number of neighbors
    % pred: predicted labels [n_test x 1]
    
    n_test = size(X_test, 1);
    pred = zeros(n_test, 1);
    
    for i = 1:n_test
        % Compute Euclidean distances from test point to all training points
        distances = sqrt(sum((X_train - X_test(i, :)).^2, 2));
        
        % Find k nearest neighbors
        [~, idx] = sort(distances);
        k_nearest = idx(1:min(k, length(idx)));
        
        % Get labels of k nearest neighbors
        k_labels = y_train(k_nearest);
        
        % Predict by majority vote
        [unique_labels, ~, label_idx] = unique(k_labels);
        counts = accumarray(label_idx, 1);
        [~, max_idx] = max(counts);
        pred(i) = unique_labels(max_idx);
    end
end

