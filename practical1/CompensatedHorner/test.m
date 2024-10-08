##x = 1.333;
##n_values = 3:42;
##classic_errors = zeros(length(n_values), 1);
##compensated_errors = zeros(length(n_values), 1);
##condition_numbers = zeros(length(n_values), 1);
##for idx = 1:length(n_values)
##    n = n_values(idx);
##    a = poly(ones(1, n));
##    true_value = polyval(a, x);
##    %disp(['True value at x = ', num2str(x), ': ', num2str(true_value)]);
##    classic_result = classicHorner(a, x);
##    %disp(['Classic Horner result: ', num2str(classic_result)]);
##    compensated_result = CompensatedHorner(a, x);
##    %disp(['Compensated Horner result: ', num2str(compensated_result)]);
##    classic_errors(idx) = abs((classic_result - true_value) / true_value);
##    compensated_errors(idx) = abs((compensated_result - true_value) / true_value);
##    condition_numbers(idx) = condp(a, x);
##    %disp(['Condition number: ', num2str(condition_numbers(idx))]);
##end
##
##figure;
##loglog(condition_numbers, classic_errors, '-o', 'DisplayName', 'Classic Horner Scheme');
##hold on;
##loglog(condition_numbers, compensated_errors, '-x', 'DisplayName', 'Compensated Horner Scheme');
##xlabel('Conditionnement');
##ylabel('Erreur directe relative');
##legend show;
##title('Conditionnement vs Erreur directe relative');
##grid on;

x=2^29+1
[a,b]=Split(x)
