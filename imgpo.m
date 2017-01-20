function [x, fx, X_sample, F_sample, T, result] = imgpo( ...
            objfun, x_domain, Nmax, XI_max, GP, result_diplay, result_save )
% 
% OUTPUTS: 
%     x  = global optimizer 
%     fx = global optimal value f(x)
%     X_sample = sampled points 
%     F_sample = sampled values of f 
%     result = intermediate results 
%         for each iteration t, result(t,:) = [N, n (split #), fmax_hat, rho_bar, xi_max, depth_T, time(s)] 
% INPUTS:  
%     objfun = objective function (to be maximised)
%     x_domain = input domain; 
%      e.g., = [-1 3; -3 3] means that domain(f) = {(x1,x2) : -1 <= x1 <= 3 and -3 <= x2 <= 3]}
%     Nmax = maximum number of evaluations for the objective function
%     XI_max = to limit the computational time due to GP: 2^2 or 2^3 is computationally reasonable (see the NIPS paper for more detail)
% 
%     GP = Gaussian process parameters, structure with fields
%       .use = boolean flag to use GP during optimisation
%       .varsigma = function handle used to compute UCB
%           UCB = mean + GP_varsigma(M) * sigma
%           e.g., nu = 0.05; GP_varsigma = @(M) sqrt(2*log(pi^2*M^2/(12*nu))); 
%       .updates: timings for hyperparameter updates
%       .likfunc = likelihood function: e.g., @likGauss
%       .meanfunc = mean function: e.g., @meanConst
%       .covfunc = covariance function (kernel): e.g., {@covMaterniso, 5}
%       .hyp = hyper-parameters: e.g., hyp.lik = -inf; hyp.mean = 0; hyp.cov = log([1/4; 1]);
%
%      result_diplay = 1: print intermidiate results
%      result_save = 1: save intermidiate result and return as result
%
% Reference: Bayesian Optimization with Exponential Convergence (Kawaguchi 2016, arXiv:1604.01348)
% Modified by J.Hadida (jhadida@fmrib.ox.ac.uk), Jan 2017.

    h_upper = max( 100, sqrt(Nmax) );
    Nspl = 0;
    result  = [];
        
    Nsmp = 1; % #of evaluations of the objective function
    Nucb = 1; % #of times the UCB is used instead of evaluating the objective
    Ndim = size(x_domain,1);

    % initialise sampling data
    x_lower = x_domain(:,1)';
    x_upper = x_domain(:,2)';
    x_size  = x_upper - x_lower;
    x_init  = (x_lower + x_upper)/2;
    f_init  = objfun(x_init);
    
    % initilisation of the tree
    T = cell(h_upper,1);
    for i = 1:h_upper
        T{i}.x_max = [];
        T{i}.x_min = [];
        T{i}.x = [];

        T{i}.f = [];

        T{i}.leaf = [];
        T{i}.new = [];

        T{i}.node =[];
        T{i}.parent =[];

        T{i}.samp = [];
    end
    tic;

    T{1}.x_min = zeros(1,Ndim);
    T{1}.x_max = ones(1,Ndim);
    T{1}.x = 0.5 * ones(1,Ndim);
    T{1}.f = f_init;
    T{1}.leaf = 1;
    T{1}.samp = 1;
    
    Nspl = Nspl + 1;

    X_sample = x_init;
    F_sample = f_init;
    LB = f_init;

    if result_diplay == 1
        fprintf(1,'N = %d, n = %d, LB = %g \n', Nsmp, Nspl, LB);
    end
    if  result_save == 1
        result = [result; Nsmp, Nspl, LB, 0, 0, 0, toc];
    end

    for h=1:h_upper
        if size(T{h}.x,1) < 1, break, end
    end
    Tdepth = h - 1;
    
    % execution
    rho_avg = 0;
    rho_bar = 0;
    xi_max = 0;
    Niter = 0;
    LB_old = LB;
    XI = 1;
    
    while Nsmp < Nmax
        
        i_max = zeros(Tdepth,1);
        b_max = -inf * ones(Tdepth,1);
        b_hi_max = -inf;
        Niter = Niter + 1;
        
        % steps (i)-(ii)
        for h = 1:Tdepth
            
            GP_label = 1;
            while GP_label == 1
                for i=1:size(T{h}.x,1)
                    if T{h}.leaf(i) == 1
                        b_hi = T{h}.f(i);
                        if b_hi > b_hi_max
                            b_hi_max = b_hi;
                            i_max(h) = i;
                            b_max(h) = b_hi;
                        end
                    end
                end
                if i_max(h) == 0, break, end
                if T{h}.samp(i_max(h)) == 1
                    GP_label = 0;
                else
                    xsample  = x_lower + T{h}.x(i_max(h),:) .* x_size;
                    fsample  = objfun(xsample);
                    X_sample = [X_sample; xsample];
                    F_sample = [F_sample; fsample];
                    
                    T{h}.samp(i_max(h)) = 1;
                    Nsmp = Nsmp+1;
                    LB = max(F_sample);
                    if result_save == 1
                        result = [result; Nsmp, Nspl, LB, rho_bar, xi_max, Tdepth, toc];
                    end
                end
            end
            
        end

        % steps (iii)
        if GP.use
            
            for h=1:Tdepth
                if i_max(h) ~= 0
                    
                    % compute xi
                    xi = 0;
                    for h_2 = h + 1 : min(Tdepth, h + min(ceil(XI),XI_max))
                        if i_max(h_2) ~= 0
                            xi = h_2 - h;
                            break;
                        end
                    end
                    if xi == 0, continue; end
                    
                    % compute z_max = z(h,i^*_h)
                    z_max = -inf;

                    for h_2 = h : h + xi
                        T2{h_2}.x_max = [];
                        T2{h_2}.x_min = [];
                        T2{h_2}.x = [];
                    end
                    T2{h}.x_max(1,:) = T{h}.x_max(i_max(h),:);
                    T2{h}.x_min(1,:) = T{h}.x_min(i_max(h),:);
                    T2{h}.x(1,:) = T{h}.x(i_max(h),:);

                    % compute z_max by expanding GP tree
                    M = Nucb;
                    for h_2 = h : h+xi-1
                        for j = 1:3^(h_2-h)

                            xx  = T2{h_2}.x(j,:);
                            x_g = xx;
                            x_d = xx;

                            [~,splitd]  = max(T2{h_2}.x_max(j,:) - T2{h_2}.x_min(j,:));
                            x_g(splitd) = (5 * T2{h_2}.x_min(j,splitd) + T2{h_2}.x_max(j,splitd))/6.0;
                            x_d(splitd) = (T2{h_2}.x_min(j,splitd) + 5 * T2{h_2}.x_max(j,splitd))/6.0;

                            xxx_g = x_lower + x_g .* x_size;
                            xxx_d = x_lower + x_d .* x_size;

                            [m_g, s2_g, GP] = gp_call( GP, X_sample, F_sample, xxx_g );
                            z_max = max(z_max, m_g+GP.varsigma(M)*sqrt(s2_g));
                            M = M + 1;

                            [m_d, s2_d, GP] = gp_call( GP, X_sample, F_sample, xxx_d );
                            z_max = max(z_max, m_d+GP.varsigma(M)*sqrt(s2_d));
                            M = M + 1;

                            if z_max >= b_max(h+xi), break, end

                            newmin = T2{h_2}.x_min(j,:);
                            newmax = T2{h_2}.x_max(j,:);
                            newmax(splitd) = (2*T2{h_2}.x_min(j,splitd)+T2{h_2}.x_max(j,splitd))/3.0;

                            T2{h_2+1}.x     = [T2{h_2+1}.x;x_g];
                            T2{h_2+1}.x_min = [T2{h_2+1}.x_min; newmin];
                            T2{h_2+1}.x_max = [T2{h_2+1}.x_max; newmax];


                            newmax = T2{h_2}.x_max(j,:);
                            newmin = T2{h_2}.x_min(j,:);
                            newmin(splitd) = (T2{h_2}.x_min(j,splitd)+2*T2{h_2}.x_max(j,splitd))/3.0;

                            T2{h_2+1}.x     = [T2{h_2+1}.x;x_d];
                            T2{h_2+1}.x_max = [T2{h_2+1}.x_max; newmax];
                            T2{h_2+1}.x_min = [T2{h_2+1}.x_min; newmin];


                            newmin = T2{h_2}.x_min(j,:);
                            newmax = T2{h_2}.x_max(j,:);
                            newmin(splitd) = (2*T2{h_2}.x_min(j,splitd)+T2{h_2}.x_max(j,splitd))/3.0;
                            newmax(splitd) = (T2{h_2}.x_min(j,splitd)+2*T2{h_2}.x_max(j,splitd))/3.0;

                            T2{h_2+1}.x     = [T2{h_2+1}.x;xx];
                            T2{h_2+1}.x_min = [T2{h_2+1}.x_min; newmin];
                            T2{h_2+1}.x_max = [T2{h_2+1}.x_max; newmax];

                        end
                        if z_max >= b_max(h+xi), break, end
                    end

                    if z_max < b_max(h+xi)
                        Nucb = M;     % if we actually used M_2 UCBs, we update M = M_2;
                        i_max(h) = 0;   % if it turns out that some UCB exceeded b_max(h+xi), then it does not matter whether or not f <= UCB. It may be UCB<f, and still it works exactly same. So, we do not have to update M in this case.
                        xi_max = max(xi,xi_max);
                    end
                end
            end
        end
        
        % steps (iv)-(v)
        b_hi_max = -inf;
        rho_t = 0;
        
        for h=1:Tdepth
            if (i_max(h) ~= 0) && (b_max(h) > b_hi_max) % JH: should this be >= (algorithm line 27)?
                
                rho_t = rho_t + 1;
                Tdepth = max(Tdepth,h+1);
                Nspl = Nspl + 1;

                T{h}.leaf(i_max(h)) = 0;
                xx = T{h}.x(i_max(h),:);

                % --- find the dimension to split:  one with the largest range ---

                x_g = xx;
                x_d = xx;
                
                [~,splitd]  = max(T{h}.x_max(i_max(h),:) - T{h}.x_min(i_max(h),:));
                x_g(splitd) = (5 * T{h}.x_min(i_max(h),splitd) + T{h}.x_max(i_max(h),splitd))/6.0;
                x_d(splitd) = (T{h}.x_min(i_max(h),splitd) + 5 * T{h}.x_max(i_max(h),splitd))/6.0;

                % --- splits the leaf of the tree ----
                
                Tmin = T{h}.x_min(i_max(h),:);
                Tmax = T{h}.x_max(i_max(h),:);
                
                % left node
                T{h+1}.x = [T{h+1}.x; x_g];
                xsample_g = x_lower + x_g .* x_size;
                UCB = +inf;
                if GP.use
                    [m, s2, GP] = gp_call( GP, X_sample, F_sample, xsample_g );
                    UCB = m+(GP.varsigma(Nucb)+0.2)*sqrt(s2);
                end
                if UCB <= LB && GP.use
                    Nucb = Nucb + 1; % need to update Nucb only if we require f <= UCB. In the other case, f can be f > UCB and not require to take union bound.
                    fsample_g = UCB;
                    T{h+1}.samp = [T{h+1}.samp 0];
                else
                    fsample_g = objfun(xsample_g);
                    T{h+1}.samp = [T{h+1}.samp 1];

                    X_sample = [X_sample; xsample_g];
                    F_sample = [F_sample; fsample_g];
                    Nsmp = Nsmp+1;
                    b_hi_max = max(b_hi_max, fsample_g);
                    LB = max(F_sample);
                    if result_save == 1
                        result = [result; Nsmp, Nspl, LB, rho_bar, xi_max, Tdepth, toc];
                    end
                end
                T{h+1}.f = [T{h+1}.f fsample_g];

                newmax = Tmax;
                newmax(splitd) = (2*Tmin(splitd)+Tmax(splitd))/3.0;
                
                T{h+1}.x_min = [T{h+1}.x_min; Tmin];
                T{h+1}.x_max = [T{h+1}.x_max; newmax];
                T{h+1}.leaf  = [T{h+1}.leaf 1];

                % right node
                T{h+1}.x = [T{h+1}.x; x_d];
                xsample_d = x_lower + x_d .* x_size;
                UCB = +inf;
                if GP.use
                    [m, s2, GP] = gp_call( GP, X_sample, F_sample, xsample_d );
                    UCB = m+(GP.varsigma(Nucb)+0.2)*sqrt(s2);
                end
                if UCB <= LB && GP.use
                    Nucb = Nucb + 1;
                    fsample_d = UCB;
                    T{h+1}.samp = [T{h+1}.samp 0];
                else
                    fsample_d = objfun(xsample_d);
                    T{h+1}.samp = [T{h+1}.samp 1];

                    X_sample = [X_sample; xsample_d];
                    F_sample = [F_sample; fsample_d];
                    Nsmp = Nsmp+1;
                    b_hi_max = max(b_hi_max, fsample_d);

                    LB = max(F_sample);
                    if result_save == 1
                        result = [result; Nsmp, Nspl, LB, rho_bar, xi_max, Tdepth, toc];
                    end
                end
                T{h+1}.f = [T{h+1}.f fsample_d];

                newmin = Tmin;
                newmin(splitd) = (Tmin(splitd)+2*Tmax(splitd))/3.0;
                
                T{h+1}.x_max = [T{h+1}.x_max; Tmax];
                T{h+1}.x_min = [T{h+1}.x_min; newmin];
                T{h+1}.leaf  = [T{h+1}.leaf 1];

                % central node
                newmin = Tmin;
                newmax = Tmax;
                newmin(splitd) = (2*Tmin(splitd)+Tmax(splitd))/3.0;
                newmax(splitd) = (Tmin(splitd)+2*Tmax(splitd))/3.0;
                
                T{h+1}.x     = [T{h+1}.x; xx];
                T{h+1}.f     = [T{h+1}.f, T{h}.f(i_max(h))];
                T{h+1}.samp  = [T{h+1}.samp 1]; % JH: 1 only with N-ary splits with N odd
                T{h+1}.x_min = [T{h+1}.x_min; newmin];
                T{h+1}.x_max = [T{h+1}.x_max; newmax];
                T{h+1}.leaf  = [T{h+1}.leaf 1];


                % --- output results -------------------------------------------------------
                LB = max(F_sample);
                if result_diplay == 1
                    fprintf('%d, N = %d, n = %d, fmax_hat = %g, rho = %d, xi = %d,h = %d, time = %f \n', ...
                        Niter, Nsmp, Nspl, LB, rho_bar, xi_max, Tdepth, toc);
                end

            end
        end
        rho_avg = (rho_avg * (Niter - 1) + rho_t) / Niter;
        rho_bar = max(rho_bar,rho_avg);

        % update Xi
        if LB_old == LB
            XI = max(XI - 2^-1,1);
        else
            XI = XI + 2^2;
        end
        LB_old = LB;

        % update GP hyper parameters
        if GP.use && ismember( Nspl, GP.updates ) % JH: we might miss an update, because Nspl can increase several units at once
            %warning ('off','all');
            GP.hyp = minimize( GP.hyp, @gp, -100, @infExact, GP.meanfunc, GP.covfunc, GP.likfunc, X_sample, F_sample);
            
%             fprintf( '--------------- Iteration #%d\n', Niter );
%             GP.hyp.cov
%             [X_sample, F_sample(:)]
%             fprintf( '---------------\n' );
            %warning ('on','all');
        end

    end

    % get a maximum point
    f_hi_max = -inf;
    for h=1:h_upper
        if size(T{h}.x,1) < 1, break, end
        for i=1:size(T{h}.x,1)
            f_hi = T{h}.f(i);
            if (f_hi > f_hi_max) % JH: should we make sure it is not GP-based too?
                f_hi_max = f_hi;
                i_max = i;
                h_max = h;
            end
        end
    end
    h=h-1;
    if result_diplay
        fprintf(1,'constructed a tree with depth h = %f \n', h);
    end

    x = x_lower + T{h_max}.x(i_max,:) .* x_size;
    fx = f_hi_max;

end

% GP prediction, make sure std of Gaussian likelihood is large enough
function [m,s2,GP] = gp_call( GP, X_sample, F_sample, x )

    hyp = GP.hyp;
    error = true;
    
    while error
        error = false;
        try
           [m,s2] = gp( hyp, @infExact, GP.meanfunc, GP.covfunc, GP.likfunc, X_sample, F_sample, x );
        catch
           error = true;
           if hyp.lik == -inf, hyp.lik = -9; end
           hyp.lik = hyp.lik + 1;
        end
    end
    GP.hyp = hyp; % JH: remember changes?

end

