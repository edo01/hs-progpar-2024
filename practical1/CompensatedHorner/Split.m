% TO SOBSTITUTE: The Split algorithm splits a floating-point number a
% into two non-overlapping parts: an high-precision x and a low-precision part y,
% such that a = x + y, where |y| <= |x|.
% We will use it in the two product algorithm to split the product in order
% to compute both the main product and the error term precisely.
% u = 2^{-p}, s = ⌈p/2⌉
function [x,y] = Split(a)
    s = 27;
    factor = 2^s+1; 
    c = factor*a; 
    x = c-(c-a);
    y = a-x;
end