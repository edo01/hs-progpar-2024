% TO ADD: The classic Horner's method evaluates the polynomial using a recursive approach.
% We saw that the classic Horner's method introduces roundings errors due to the
% multiplication and the addition operations. The associated condition number 
% is $cond(p,x)=\sum_{i=0}^n |a_i| |x|^i / |p(x)|$.
% So using the rule of thumb, we can expect an estimated relative error of
% $cond(p,x) * u$, where $u$ is the unit roundoff. So, for big condition numbers,
% e.g. when the value of the polynomial is near zero, the error will be big
% depending on the precision of the floating-point number.
function res = classicHorner(a, x)
    n = length(a);
    res = a(n);
    for i = n-1:-1:1
        % every iteration we introduce roundings errors
        % due to the multiplication and the addition
        res = res * x + a(i); 
    end
end