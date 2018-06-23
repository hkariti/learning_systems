function [ output ] = DT_Classify(tree, test_set)
% Classify Test set
output = zeros(1, length(test_set));
for i = 1:length(test_set)
    cur_node = tree;
    while cur_node.leaf == false
        if test_set(cur_node.feature, i) <= cur_node.threshold
            cur_node = cur_node.left;
        else
            cur_node = cur_node.right;
        end
    end
    output(i) = cur_node.class;
end
end

