function [x, y] = TwoProduct(a, b)
    x = a * b; % main product
    [a1, a2] = Split(a);
    [b1, b2] = Split(b);
    y = a2 * b2 - (((x - a1 * b1) - a2 * b1) - a1 * b2); % error
end