function [x, fx, X_sample, F_sample, tree] = imgpo_default( objfun, x_domain, nb_eval, verbose )
%
% execute imgpo with a default setting used in a NIPS paper
%
% output: 
%      x = global optimizer 
%      fx = global optimal value f(x)
% optinal output:
%      X_sample = sampled points 
%      F_sample = sampled values of f 
%      result = intermidiate results 
%               for each iteration t, result(t,:) = [N, n (split #), fmax_hat, rho_bar, xi_max, depth_T, time(s)] 
% input: 
%      objfun = ovjective function (to be optimized)
%      x_input_domain = input domain; 
%          e.g., = [-1 3; -3 3] means that domain(f) = {(x1,x2) : -1 <= x1 <= 3 and -3 <= x2 <= 3]} 
%      nb_eval = the number of evaluations allowed
% input display flag:
%      result_diplay = 1: print intermidiate results
%      result_save = 1: save intermidiate result and return as result

    % ------- parameter for the main algorithm ------- 
    % for low dimension
    XI_max = 2^2;     % to limit the computational time due to GP: 2^2 or 2^3 is computationally reasonable (see the NIPS paper for more detail)
    %XI_max = 2^3;    % lower this if it is too slow 

    % ------- parameters for GP ------- 
    GP_use = 1;        % = 1: use GP
    nu = 0.05;         % theoretical gurantee holds with probability 1 - nu 
    GP_varsigma = @(M) sqrt(2*log(pi^2*M^2/(12*nu)));  % UCB = mean + GP_varsigma(M) * sigma

    GP_updates = ... % = the timing of updating hyper_parameters during execusion: modify this to save computational time  
        [1,2,3,4,5,floor(logspace(1,5))]; 
    %GP_kernel_est_timing = 1:1:nb_iter; % the setting used in the NIPS paper experiment to be fair with previous methods. But, this is not practical

    % ------- parameters for gpml library (see the manual of gpml library for detail) -------
    likfunc  = @likGauss;  % likelihood function
    meanfunc = @meanConst; %  mean function
    covfunc  = {@covMaterniso, 5}; % covariance function (kernel)

    ell = 1/4; sf = 1; sigma = 1e-6;
    hyp.lik  = log(sigma); 
    hyp.mean = 0; 
    hyp.cov  = log([ell; sf]); % hyper-parameters

    GP.use = GP_use;
    GP.varsigma = GP_varsigma;
    GP.updates = GP_updates;
    GP.likfunc = likfunc;
    GP.meanfunc = meanfunc;
    GP.covfunc = covfunc;
    GP.hyp = hyp;

    [x, fx, X_sample, F_sample, tree] = imgpo( ...
        objfun, x_domain, nb_eval, XI_max, GP, verbose, 0 );

end
