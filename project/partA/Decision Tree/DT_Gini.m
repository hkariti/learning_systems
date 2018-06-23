function [ Q ] = DT_Gini(p)
    Q = sum(p .* (1 - p));
end

