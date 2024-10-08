function res = exactHorner(a, x)
    syms xs; % Define a symbolic variable
    poly = poly2sym(a, xs); % Create a symbolic polynomial using the coefficients 'a'
    res = double(subs(poly, xs, x)); % Substitute 'x' and convert the result to a double (numeric)
end
