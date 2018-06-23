function [ tree ] = DT_Train(samples, tags, measure_unity)

if isempty(tags)
    tree = struct;
    tree.leaf = true;
    tree.class = 1;
    return;
end

if all(tags == 1) || all(tags == -1)
    tree = struct;
    tree.leaf = true;
    tree.class = tags(1);
    return;
end

[opt_feature, opt_threshold] = DT_Find_Opt_Feature(samples, tags, measure_unity);
under_threshold_idx = (samples(opt_feature,:) <= opt_threshold);

samples_g1 = samples(:, under_threshold_idx);
tags_g1 = tags(under_threshold_idx);
samples_g2 = samples(:, ~under_threshold_idx);
tags_g2 = tags(~under_threshold_idx);

tree = struct;
tree.feature = opt_feature;
tree.threshold = opt_threshold;
tree.leaf = false;
tree.left = DT_Train(samples_g1, tags_g1, measure_unity);
tree.right = DT_Train(samples_g2, tags_g2, measure_unity);
end