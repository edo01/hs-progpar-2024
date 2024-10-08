function cond_num = condp(a, x)
    n = length(a) - 1;
    abs_sum = sum(abs(a) .* abs(x).^[0:n]);
    p_val = polyval(a, x);
    cond_num = abs_sum / abs(p_val);
end