%testHornerSchemes()
format longE;
num_datasets = 10;
studyAlgorithmAccuracy(num_datasets);
function studyAlgorithmAccuracy(num_datasets) 
    condition_numbers = logspace(0, 4, num_datasets); 
    relative_errors = zeros(num_datasets, 4);  
    n = 1000; 
    for i = 1:num_datasets
        p = randn(n, 1) * condition_numbers(i);
        vrai_sum = sum(p);
        
        fast = SCompSum(p);  % SCompSum 
        kahan = Sum(p);  %  Sum
        priest = DCompSum(p);  % DCompSum 
        rump = CompSum(p);  % CompSum 

       
        relative_errors(i, 1) = abs(fast - vrai_sum) / abs(vrai_sum);
        relative_errors(i, 2) = abs(kahan - vrai_sum) / abs(vrai_sum);
        relative_errors(i, 3) = abs(priest - vrai_sum) / abs(vrai_sum);
        relative_errors(i, 4) = abs(rump - vrai_sum) / abs(vrai_sum);
    end

    
    figure;
    loglog(condition_numbers, relative_errors,'LineWidth', 2);
    %plot(condition_numbers, relative_errors,'LineWidth', 2);
    xlabel('(Condition Number)');
    ylabel('(Relative Error)');
    legend('SCompSum', 'Sum', 'DCompSum', 'CompSum', 'Location', 'Best');
    title('Accuracy relative to condition number');
    grid on;
end

% Function to test Classic and Compensated Horner schemes
% This function computes and plots the direct relative errors
% and condition numbers for the polynomials p_n(x) = (x - 1)^n

function testHornerSchemes()
    x = 1.333; 
    n_values = 3:42; 
    classic_errors = zeros(length(n_values), 1); 
    compensated_errors = zeros(length(n_values), 1);
    condition_numbers = zeros(length(n_values), 1); 
    for idx = 1:length(n_values)
        n = n_values(idx);
        a = poly(ones(1, n)); 
        true_value = polyval(a, x);
        %disp(['True value at x = ', num2str(x), ': ', num2str(true_value)]); 
        classic_result = classicHorner(a, x);
        %disp(['Classic Horner result: ', num2str(classic_result)]); 
        compensated_result = CompensatedHorner(a, x);
        %disp(['Compensated Horner result: ', num2str(compensated_result)]);
        classic_errors(idx) = abs((classic_result - true_value) / true_value);
        compensated_errors(idx) = abs((compensated_result - true_value) / true_value);
        condition_numbers(idx) = condp(a, x);
        %disp(['Condition number: ', num2str(condition_numbers(idx))]); 
    end

    figure;
    loglog(condition_numbers, classic_errors, '-o', 'DisplayName', 'Classic Horner Scheme');
    hold on;
    loglog(condition_numbers, compensated_errors, '-x', 'DisplayName', 'Compensated Horner Scheme');
    xlabel('Conditionnement');
    ylabel('Erreur directe relative'); 
    legend show;
    title('Conditionnement vs Erreur directe relative');
    grid on;
end


function res = Sum(p)
    sigma = 0;
    for i = 1:length(p)
        sigma = sigma + p(i);
    end
    res = sigma;
end

function res = SCompSum(p)
    sigma = 0;
    e = 0;
    for i = 1:length(p)
        y = p(i) + e;
        [sigma, e] = FastTwoSum(sigma, y);
    end
    res = sigma;
end

function res = DCompSum(p)
    n = length(p);
    [sorted_p, idx] = sort(abs(p), 'descend');
    sorted_p = p(idx);
    s = 0;
    c = 0;
    for i = 1:n
        [y, u] = FastTwoSum(c, sorted_p(i));
        [t, v] = FastTwoSum(y, s);
        z = u + v;
        [s, c] = FastTwoSum(t, z);
    end
    res = s;
end


function res = CompSum(p)
    pi = p(1); sigma = 0;
    for i = 2:length(p)
        [pi, qi] = TwoSum(pi, p(i));
        sigma = sigma + qi;
    end
    res = pi + sigma;
end
