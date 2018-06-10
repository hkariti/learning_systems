function [ format, nodes_info ] = DT_Export(tree)
% export to matlab treeplot format
if tree.leaf
    format = 0;
    if tree.class == -1
        tree.class = 0;
    end
    nodes_info = {tree};
    return;
end
[format_left, nodes_info_left] = DT_Export(tree.left);
[format_right, nodes_info_right] = DT_Export(tree.right);
format_right(2:end) = format_right(2:end) + length(format_left);
format = [0, format_left+1, format_right+1];
nodes_info = {tree, nodes_info_left{:}, nodes_info_right{:}};
end

