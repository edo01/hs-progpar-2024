% TO ADD: This function requires |a|>|b|. It requires less operations than TwoSum but 
% introduces a restriction on the domain of the inputs.
function [x, y] = FastTwoSum(a, b)
    x = a + b;
    y = (a - x) + b;
end