function [ train_samples_sets, train_tags_sets, test_samples_sets, test_tags_sets ] = cross_validation(train_samples, train_tags, num_of_sets)
% returns cross-validation data consists of 10 sets of test's data and tags
% and train's data and tags
[num_of_samples, num_of_features] = size(train_samples);
test_set_size = floor(num_of_samples / num_of_sets);
train_set_size = num_of_samples - test_set_size;
indexes = randperm(num_of_samples);
train_samples = train_samples(indexes, :);
train_tags = train_tags(indexes);
train_samples_sets = zeros(num_of_sets, train_set_size, num_of_features);
train_tags_sets = zeros(num_of_sets, train_set_size);
test_samples_sets = zeros(num_of_sets, test_set_size, num_of_features);
test_tags_sets = zeros(num_of_sets, test_set_size);
for i = 1 : num_of_sets
    test_start_idx = (i - 1) * test_set_size + 1;
    test_end_idx = i * test_set_size;
    test_samples_sets(i,:,:) = train_samples(test_start_idx:test_end_idx, :);
    test_tags_sets(i,:) = train_tags(test_start_idx:test_end_idx, :);
    temp = train_samples;
    temp(test_start_idx:test_end_idx, :) = [];
    train_samples_sets(i,:,:) = temp;
    temp = train_tags;
    temp(test_start_idx:test_end_idx, :) = [];
    train_tags_sets(i,:,:) = temp;
end
save('cross_validation_sets.mat', 'train_samples_sets', 'train_tags_sets', 'test_samples_sets', 'test_tags_sets');
end