function [ Q ] = DT_Entropy(p)
    Q = -sum(p .* log2(p));
    if isnan(Q)
        Q = 0;
    end
end

