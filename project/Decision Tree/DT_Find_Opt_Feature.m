function [opt_feature, opt_threshold] = DT_Find_Opt_Feature(samples, tags, measure_unity)

% Calculating relativity ditribution of tags & Q (non unify measurement)
p = DT_Histogram(tags) ./ length(tags);
Q = measure_unity(p);

opt_info_gain = -Inf;
[num_of_features, num_of_samples] = size(samples);
% Iterate through all features
for feature = 1:num_of_features
    samples_feature = samples(feature,:);
    % Find Optimal threshold for this feature
    for i = 1:num_of_samples
        threshold = samples_feature(i);
        tags_g1 = tags(samples_feature <= threshold);
        p_g1 = DT_Histogram(tags_g1) ./ length(tags_g1);
        tags_g2 = tags( ~(samples_feature <= threshold) );
        p_g2 = DT_Histogram(tags_g2) ./ length(tags_g2);
        Q_i = (length(tags_g1) * measure_unity(p_g1) + length(tags_g2) * measure_unity(p_g2)) / length(tags);
        gain = Q - Q_i;
        if gain > opt_info_gain
            opt_threshold = threshold;
            opt_feature = feature;
            opt_info_gain = gain;
        end
    end
end
end

