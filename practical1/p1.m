% we notice that the error is around epsilon machine
% almost e-16 for the double precision and
% and e-8 for the single precision

%a = single(1.2345);
%b = single(0.5678);
a = 1.2345;
b = 0.5678;
[x, y] = FastTwoSum(a, b);
[x_ef, y_ef] = TwoSum(a, b);

disp('FastTwoSum：');
disp(['x = ', num2str(x)]);
disp(['y = ', num2str(y)]);
disp('TwoSum ：');
disp(['x = ', num2str(x_ef)]);
disp(['y = ', num2str(y_ef)]);

normal_sum = a + b;
disp("The result of a normal sum:");
disp(['sum = ', num2str(normal_sum)]);

p = [ 2 3 5 6];
x = 0.56;
res   = polyval(flip(p), x)
res_c =  Horner(p, x)

disp("Horner:");
disp("res_c =", res_c);

disp("Polyval:");
disp("res =", res);


% p is an array of the polynomials coefficients
function res = Horner(p, x)
  n = length(p); 
  s(n) = p(end); % s_n = a_n
  for i=n-1:-1:1
      prod = s(i+1)*x;
      s(i) = prod+p(i);
  end
  res = s(1);
end

function [x, y] = FastTwoSum(a, b)
    x = a + b;
    y = (a - x) + b;
end

function [x, y] = TwoSum(a, b)
    x = a + b;
    z = x - a;
    y = (a - (x - z)) + (b - z);
end

function [x,y] = Split(a)
    s = 27;
    factor = 2^s+1; % s = 27
    c = factor*a;
    x = c-(c-a);
    y = a-x;
end

function [x, y] = TwoProduct(a,b);
    x = a*b;
    [a1, a2] = Split(a);
    [b1, b2] = Split(b);
    y = a2*b2-(((x-a1*b1)))

