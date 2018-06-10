function [ p ] = DT_Histogram(tags)
% Histogram of tags
if isempty(tags)
    p = zeros(1, 2);
else
    p = [sum(tags == -1), sum(tags == 1)];
end
end

