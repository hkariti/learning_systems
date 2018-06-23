function [height] = DT_Height(tree)
if tree.leaf
    height = 0;
    return;
end
left_height = DT_Height(tree.left);
right_height = DT_Height(tree.right);
height = max(left_height, right_height) + 1;
end