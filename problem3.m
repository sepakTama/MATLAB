
n = 100;
x = 3 * (rand(n, 2) - 0.5);
radius = x(:, 1).^2 + x(:, 2).^2;
y = (radius > 0.7 + 0.1 * randn(n, 1)) & (radius < 2.2 + 0.1 * randn(n, 1));
y = 2 * y -1;


%{
n = 40;
omega = randn(1, 1);
noise = 0.8 * randn(n, 1);
x = randn(n, 2);
y = 2 * (omega * x(:, 1) + x(:, 2) + noise > 0) - 1;
%}

hold on

%scatter(x(y(:) == -1, 1),x(y(:) == -1, 2) , 25, 'r', 'filled');
%scatter(x(y(:) ~= -1, 1),x(y(:) ~= -1, 2) , 25, 'b', 'filled');

dual_gap = projectedGradient(x, y);

hold off

function [dual_gap] = projectedGradient(x_mat, y_vec)
% implementation of profected gradient method

    [n, ~] = size(x_mat);
    alpha_t1 = ones(n, 1);
    K = zeros(n, n);
    for i = 1:n
        for j = 1:n
            K(i,j) = y_vec(i) * y_vec(j) * dot(x_mat(i, :), x_mat(j, :));
        end
    end

    gamma = max(eig(2*K));
    eta = 1 / gamma;
    vec_1 = ones(size(alpha_t1));
    iter = 1;
    lambda = 0.5;

    while iter <= 100
        w_hat = zeros(size(x_mat(1, :)));
        prime_value = 0;
        alpha_t = alpha_t1;
        
        alpha_t1 = mapping(alpha_t - eta * ((K * alpha_t) / (2 * lambda) - vec_1));
        % calculate prime value
        for i = 1:n
            w_hat = w_hat + alpha_t1(i) * y_vec(i) * x_mat(i, :);
        end
        w_hat = w_hat / (2 * lambda);
        for i = 1:size(y_vec, 1)
            prime_value = prime_value + max(0, 1- y_vec(i) * dot(w_hat, x_mat(i, :)));
        end
        prime_value = prime_value + lambda * (norm(w_hat))^2;   
        % calculate dual value
        dual_value = -(alpha_t1' * K * alpha_t1) / (4 * lambda) + dot(alpha_t1, vec_1);
        
        dual_gap = abs(prime_value - dual_value);       
        
        scatter(iter, prime_value, 25, 'r', 'filled');
        scatter(iter, dual_value, 25, 'b', 'filled');
        %scatter(iter, dual_gap, 25, 'k');
        iter = iter + 1;
    end
    %fplot(@(x) -(w_hat(1)/w_hat(2))*x, 'k');
end


function [x] = mapping(x)
% mapping to [0,1]
x(x(:) > 1) = 1;
x(x(:) < 0) = 0;
end

