%% Prepare Sets
load('BreastCancerData.mat');

test_indices = [3,11,17,21,22,28,29,41,44,46,68,69,70,77,84,95,116,124,127,130,133,139,144,145,147,162,166,168,173,183,184,185,188,195,198,208,212,213,215,221,222,226,227,230,234,243,244,247,252,256,257,259,261,263,266,271,275,276,285,292,293,297,298,307,309,310,313,318,321,326,337,345,346,356,358,362,363,364,374,377,378,379,381,389,414,416,423,438,443,446,456,462,465,467,469,477,478,480,486,489,492,493,505,514,516,518,519,520,523,524,525,538,549,553,568];
test_set_idx = zeros(length(y), 1);
test_set_idx(test_indices) = 1;
training_set_idx = ~test_set_idx;
test_set_idx = ~training_set_idx;
train_samples = X(:,training_set_idx)';
train_tags = y(training_set_idx);
tests_samples = X(:,test_set_idx)';
tests_tags = y(test_set_idx);

% Cross-Validate
num_of_sets = 10;
%[train_samples_sets, train_tags_sets, test_samples_sets, test_tags_sets] = Cross_Validate( train_samples, train_tags, num_of_sets );
load('cross_validation_sets.mat')
tests_samples = tests_samples';
train_samples = train_samples';
train_tags_sets(train_tags_sets<1) = -1;
test_tags_sets(test_tags_sets<1) = -1;
train_tags(train_tags<1) = -1;
tests_tags(tests_tags<1) = -1;

%% Training and Testing Decision Tree with Cross-Validation and 3 Measuring Techniques
tic
error = zeros(3, num_of_sets);
error_mean = zeros(3, 1);
error_std = zeros(3, 1);
measurement = {'Classification Error', @DT_Class_Error;
    'Gini Index', @DT_Gini;
    'Entropy', @DT_Entropy;};
min_error = Inf;
for j = 1:length(measurement)
    error_vec = zeros(num_of_sets, 1);
    for i = 1:num_of_sets
        train_samples_cross_i = transpose(squeeze(train_samples_sets(i,:,:)));
        train_tags_cross_i = train_tags_sets(i,:);
        test_samples_cross_i = transpose(squeeze(test_samples_sets(i,:, :)));
        test_tags_cross_i = test_tags_sets(i,:);
        tree = DT_Train(train_samples_cross_i, train_tags_cross_i, measurement{j, 2});
        output = DT_Classify(tree, test_samples_cross_i);
        error_vec(i) = 1 - sum(output==test_tags_cross_i) / length(test_tags_cross_i);
        if error_vec(i) < min_error
            min_error = error_vec(i);
            opt_tree = tree;
        end
    end
    error(j,:) = error_vec;
    error_mean(j) = mean(error(j,:));
	error_std(j) = std(error(j,:));
end

figure();
errorbar(1:length(measurement), error_mean, error_std, 'X')
for i=1:length(measurement)
    text(i, error_mean(i), measurement{i,1});
end
xlim([0 5])
ylabel('Classification Error');
xlabel('Measure Method');

%% Show Parameters & Plot Decision Tree
tree = DT_Train(train_samples, train_tags, @DT_Entropy);
tree_height = DT_Height(tree);
output = DT_Classify(tree, tests_samples)';
error_test = 1 - sum(output==tests_tags) / length(tests_tags);
output = DT_Classify(tree, train_samples)';
error_train = 1 - sum(output==train_tags) / length(train_tags);
disp(['Optimal Tree Height: ' num2str(tree_height)]);
disp(['Optimal Tree Train Error Rate: ', num2str(error_train*100),'%']);
disp(['Optimal Tree Test Error Rate: ', num2str(error_test*100),'%']);
disp(['Algo Run Time: ', num2str(toc)]);

figure();
[layout, nodes_info] = DT_Export(tree);
treeplot(layout);
set(gca,'XTick',[])
set(gca,'YTick',[])
title('Optimal Decision Tree');
[nodes_x, nodes_y] = treelayout(layout);
for i=1:length(layout)
    if nodes_info{i}.leaf
        text(nodes_x(i), nodes_y(i)-0.03, num2str(nodes_info{i}.class));
    else
        text(nodes_x(i), nodes_y(i)-0.01, ['Feature ' num2str(nodes_info{i}.feature) ' <= ' num2str(nodes_info{i}.threshold)]);
    end
end
    