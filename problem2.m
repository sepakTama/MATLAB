A = [3, 0.5; 0.5, 1];
mu = [1; 2];

w_opt_2 = [0.82; 1.09];
w_opt_4 = [0.64; 0.18];
w_opt_6 = [0.33; 0];

lambda = 2;
w_opt = w_opt_2;


A = [250,15;15,4];
mu = [1;2];
lambda = 0.89;
w_opt = [1; 1.865];


hold on
w_hat_1 = proximalGradient(A, mu, lambda, w_opt);
w_hat_2 = accelerationProximal(A, mu, lambda, w_opt);
w_hat_3 = adaGrad(A, mu, lambda, w_opt);
hold off
w_hat_1
w_hat_2
w_hat_3

function [w_hat] = proximalGradient(A, mu, lambda, w_opt)
% Implementation of proximal gradient method
gamma = max(eig(2*A));
nablaPhi = @(w) 2*A*(w - mu);

w_t1 = [3;-1]; % initialize point
q = lambda /gamma;
iter = 1;
while iter <= 100
    w_t = w_t1;
    mu_tilda = w_t - nablaPhi(w_t) / gamma;
    for i = 1:size(w_t, 1)
        w_t1(i) = softThreshold(mu_tilda(i), q);
    end
    scatter(w_t1(1), w_t1(2), 15, 'r', 'filled');
    plot([w_t(1),w_t1(1)],[w_t(2), w_t1(2)], 'm');
    %scatter(iter, log(norm(w_t1 - w_opt)), 25, 'r', 'filled');
    iter = iter + 1;
end
w_hat = w_t1;
end


function [w_hat] = accelerationProximal(A, mu, lambda, w_opt)
% Implementation of proximal gradient method

gamma = max(eig(2*A));
nablaPhi = @(w) 2*A*(w - mu);
q_t = @(t) (t-1) / (t+2);

w_t1 = [3;-1]; % initialize point
v_t1 = [3;-1];
q = lambda /gamma;

iter = 1;
while iter <= 100
    w_t = w_t1;
    v_t = v_t1;
    mu_tilda = v_t - nablaPhi(v_t) / gamma;
    for i = 1:size(w_t, 1)
        w_t1(i) = softThreshold(mu_tilda(i), q);
    end
    v_t1 = w_t1 + q_t(iter+1)*(w_t1 - w_t);
    scatter(w_t1(1), w_t1(2), 15, 'b', 'filled');
    plot([w_t(1),w_t1(1)],[w_t(2), w_t1(2)], 'c');
    %scatter(iter, log(norm(w_t1 - w_opt)), 25, 'b', 'filled');
    iter = iter + 1;
end
w_hat = w_t1;
end


function[w_hat] = adaGrad(A, mu, lambda, w_opt)
% implementation of AdaGrad optimization method

gamma = max(eig(2*A));
delta = 0.02;
nablaPhi = @(w) 2*A*(w - mu);
eta = 500 / gamma;

w_t1 = [3;-1]; % initialize point
iter = 1;
g_mat = [];

while iter <= 100
    w_t = w_t1;
    
    g_mat = [g_mat, nablaPhi(w_t1)];
    
    for i = 1:size(w_t, 1)
        g_t1 = g_mat(i,:);
        H_t1 = sqrt(sum(g_t1.^2)) + delta;
        mu_tilda_i = w_t(i) - (eta * g_mat(i, iter))/H_t1;
        q = (eta * lambda) / H_t1;
        w_t1(i) = softThreshold(mu_tilda_i, q);
    end
    scatter(w_t1(1), w_t1(2), 15, 'g', 'filled');
    plot([w_t(1),w_t1(1)],[w_t(2), w_t1(2)], 'k');
    %scatter(iter, log(norm(w_t1 - w_opt)), 25, 'g', 'filled');
    iter = iter + 1;
end
w_hat = w_t1;
size(w_t, 1)
end




function [w_t1_i] = softThreshold(mu_i, q)
    if mu_i > q
        w_t1_i = mu_i - q;
    elseif mu_i < -q
        w_t1_i = mu_i + q;
    else
        w_t1_i = 0;
    end
end

