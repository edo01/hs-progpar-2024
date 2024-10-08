% The compensated Horner's method evaluates the polynomial using EFT (Error-Free Transformation)
% to take into account the roundings errors. At each iteration, we compute the EFT product and
% the EFT sum, and we accumulate the error term.
% 
% \[p(x) = Horner(p, x) + (p\pi + p\sigma)(x)\]
% where p\pi and p\sigma are the polynomial obtained using as coefficients the error terms of the
% product and the sum respectively.
% 
% This scheme leads to a more accurate result than the classic Horner's method, since the relative
% error is bounded by the quantity $u + \gamma_{2n}cond(p,x)$, where $\gamma_{2n}$ is about $4n^2*u^2$.
% Now, the condition number is multiplied by $u^2$, which allows to reduce the error for big condition 
% numbers.
function res = CompensatedHorner(a, x)
    n = length(a);
    s = a(n);
    r = 0; % r represents the error
    for i = n-1:-1:1
		% Calculate the product and error of the current item
        [p, pi] = TwoProduct(s, x); 
        [s, sigma] = TwoSum(p, a(i));
        r = r * x + (pi + sigma);
    end
    res = s + r;
end