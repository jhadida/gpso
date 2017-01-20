function [x, fx, X_sample, F_sample, T, result] = imgpo( ...
            objfun, x_domain, Nmax, XI_max, ...
            GP_use, GP_updates, GP_varsigma, likfunc, meanfunc, covfunc, hyp, ...
            result_diplay, result_save )
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
%      x_domain = input domain; 
%          e.g., = [-1 3; -3 3] means that domain(f) = {(x1,x2) : -1 <= x1 <= 3 and -3 <= x2 <= 3]}
%      nb_iter = the number of iterations to perform
% input: algorithm  
%      XI_max = to limit the computational time due to GP: 2^2 or 2^3 is computationally reasonable (see the NIPS paper for more detail)
% input: Gaussian process
%      GP_use = 1: use GP 
%      GP_kernel_est = 1: update kernel parameters  during execusion   
%      GP_varsigma: UCB = mean + GP_varsigma(M) * sigma
%                   e.g., nu = 0.05; GP_varsigma = @(M) sqrt(2*log(pi^2*M^2/(12*nu))); 
%                   e.g., GP_varsigma = @(M) 2; 
% input: gpml library (see the manual of gpml library for detail)
%      likfunc = likelihood function: e.g., @likGauss
%      meanfunc = mean function: e.g., @meanConst
%      covfunc = covariance function (kernel): e.g., {@covMaterniso, 5}
%      hyp = hyper-parameters: e.g., hyp.lik = -inf; hyp.mean = 0; hyp.cov = log([1/4; 1]);
% input display flag:
%      result_diplay = 1: print intermidiate results
%      result_save = 1: save intermidiate result and return as result


    %% initilisation
    h_upper = max( 100, sqrt(Nmax) );
    Nspl = 0;
    result = [];

    Nsmp = 0;
    Nucb = 1;
    Ndim = size(x_domain,1);

    % rescale parameters to [0,1]^D
    x_lower = x_domain(:,1)';
    x_upper = x_domain(:,2)';
    x_width = x_upper - x_lower;
    x_init  = (x_lower + x_upper)/2;
    f_init  = objfun(x_init);
    
    assert( isscalar(f_init), 'Objective function must be scalar.' );
    

    %% initilisation of the tree
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
    tic

    T{1}.x_min = zeros(1,Ndim);
    T{1}.x_max = ones(1,Ndim);
    T{1}.x = 0.5 * ones(1,Ndim);
    T{1}.f = f_init;
    T{1}.leaf = 1;
    T{1}.samp = 1;
    
    Nsmp = Nsmp + 1;
    Nspl = Nspl + 1;
    Tdepth = 1;

    X_sample = x_init;
    F_sample = f_init;
    LB = f_init;

    if result_diplay == 1
        fprintf(1,'N = %d, n = %d, LB = %g \n', Nsmp, Nspl, LB);
    end
    if  result_save == 1
        result = [result; Nsmp, Nspl, LB, 0, 0, 0, toc];
    end
    
    %% execution
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
        
        %% steps (i)-(ii)
        for h = 1:Tdepth
            
            GP_label = 1;
            while GP_label == 1
                for i=1:size(T{h}.x,1)
                    if (T{h}.leaf(i) == 1)
                        b_hi = T{h}.f(i);
                        if (b_hi > b_hi_max)
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
                    xsample = x_lower + T{h}.x(i_max(h),:) .* x_width;
                    fsample = objfun(xsample);
                    T{h}.samp(i_max(h)) = 1;
                    X_sample = [X_sample; xsample];
                    F_sample = [F_sample; fsample];
                    Nsmp = Nsmp+1;
                    if result_save == 1
                        LB = max(F_sample);
                        result = [result; Nsmp, Nspl, LB, rho_bar, xi_max, Tdepth, toc];
                    end
                end
            end
            
        end

        %% steps (iii)
        if GP_use == 1
            for h=1:Tdepth
                if(i_max(h) ~= 0)
                    % compute xi
                    xi = 0;
                    for h_2 = h + 1 : min(Tdepth, h + min(ceil(XI),XI_max))
                        if(i_max(h_2) ~= 0)
                            xi = h_2 - h;
                            break;
                        end
                    end
                    % compute z_max = z(h,i^*_h)
                    z_max = -inf;
                    if xi ~=0
                        % prepare
                        for h_2 = h : h + xi
                            t2{h_2}.x_max = [];
                            t2{h_2}.x_min = [];
                            t2{h_2}.x = [];
                        end
                        t2{h}.x_max(1,:) = T{h}.x_max(i_max(h),:);
                        t2{h}.x_min(1,:) = T{h}.x_min(i_max(h),:);
                        t2{h}.x(1,:) = T{h}.x(i_max(h),:);

                        % compute z_max by expanding GP tree
                        M_2 = Nucb;
                        for h_2 = h : h+xi-1
                            for i_2 = 1:3^(h_2-h)
                                xx = t2{h_2}.x(i_2,:);
                                [~, splitd] = max(t2{h_2}.x_max(i_2,:) - t2{h_2}.x_min(i_2,:));
                                x_g = xx;
                                x_g(splitd) = (5 * t2{h_2}.x_min(i_2,splitd) + t2{h_2}.x_max(i_2,splitd))/6.0;
                                x_d = xx;
                                x_d(splitd) = (t2{h_2}.x_min(i_2,splitd) + 5 * t2{h_2}.x_max(i_2,splitd))/6.0;
                                
                                xxx_g = x_lower + x_g .* x_width;
                                xxx_d = x_lower + x_d .* x_width;
                                
                                [m_g, s2_g] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx_g);
                                z_max = max(z_max, m_g+GP_varsigma(M_2)*sqrt(s2_g));
                                M_2 = M_2 + 1;

                                [m_d, s2_d] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx_d);
                                z_max = max(z_max, m_d+GP_varsigma(M_2)*sqrt(s2_d));
                                M_2 = M_2 + 1;

                                if z_max >= b_max(h+xi), break, end

                                t2{h_2+1}.x = [t2{h_2+1}.x;x_g];
                                newmin = t2{h_2}.x_min(i_2,:);
                                t2{h_2+1}.x_min = [t2{h_2+1}.x_min; newmin];
                                newmax = t2{h_2}.x_max(i_2,:);
                                newmax(splitd) = (2*t2{h_2}.x_min(i_2,splitd)+t2{h_2}.x_max(i_2,splitd))/3.0;
                                t2{h_2+1}.x_max = [t2{h_2+1}.x_max; newmax];

                                t2{h_2+1}.x = [t2{h_2+1}.x;x_d];
                                newmax = t2{h_2}.x_max(i_2,:);
                                t2{h_2+1}.x_max = [t2{h_2+1}.x_max; newmax];
                                newmin = t2{h_2}.x_min(i_2,:);
                                newmin(splitd) = (t2{h_2}.x_min(i_2,splitd)+2*t2{h_2}.x_max(i_2,splitd))/3.0;
                                t2{h_2+1}.x_min = [t2{h_2+1}.x_min; newmin];

                                t2{h_2+1}.x = [t2{h_2+1}.x;xx];
                                newmin = t2{h_2}.x_min(i_2,:);
                                newmax = t2{h_2}.x_max(i_2,:);
                                newmin(splitd) = (2*t2{h_2}.x_min(i_2,splitd)+t2{h_2}.x_max(i_2,splitd))/3.0;
                                newmax(splitd) = (t2{h_2}.x_min(i_2,splitd)+2*t2{h_2}.x_max(i_2,splitd))/3.0;
                                t2{h_2+1}.x_min = [t2{h_2+1}.x_min; newmin];
                                t2{h_2+1}.x_max= [t2{h_2+1}.x_max; newmax];
                            end
                            if z_max >= b_max(h+xi), break, end
                        end

                    end

                    if xi ~= 0 && z_max < b_max(h+xi)
                        Nucb = M_2;                           % if we actually used M_2 UCBs, we update M = M_2;
                        i_max(h) = 0;                      % if it turns out that some UCB exceeded b_max(h+xi), then it does not matter whether or not f <= UCB. It may be UCB<f, and still it works exactly same. So, we do not have to update M in this case.
                        xi_max = max(xi,xi_max);
                    end
                end
            end
        end
        
        %% steps (iv)-(v)
        b_hi_max_2 = -inf;
        rho_t = 0;
        for h=1:Tdepth
            if(i_max(h) ~= 0 && b_max(h) > b_hi_max_2)
                %if(i_max(h) ~= 0)
                rho_t = rho_t + 1;
                Tdepth = max(Tdepth,h+1);
                Nspl = Nspl + 1;

                T{h}.leaf(i_max(h)) = 0;

                xx = T{h}.x(i_max(h),:);

                % --- find the dimension to split:  one with the largest range ---

                [~, splitd] = max(T{h}.x_max(i_max(h),:) - T{h}.x_min(i_max(h),:));
                x_g = xx;
                x_g(splitd) = (5 * T{h}.x_min(i_max(h),splitd) + T{h}.x_max(i_max(h),splitd))/6.0;
                x_d = xx;
                x_d(splitd) = (T{h}.x_min(i_max(h),splitd) + 5 * T{h}.x_max(i_max(h),splitd))/6.0;

                % --- splits the leaf of the tree ----
                % left node
                T{h+1}.x = [T{h+1}.x;x_g];
                xxx_g = x_lower + x_g .* x_width;
                UCB = +inf;
                if GP_use == 1
                    [m, s2] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx_g);
                    UCB = m+(GP_varsigma(Nucb)+0.2)*sqrt(s2);
                end
                if UCB <= LB && GP_use == 1
                    Nucb = Nucb + 1;                         % need to update M only if we require f <= UCB. In the other case, f can be f > UCB and not require to take union bound.
                    fsample_g = UCB;
                    T{h+1}.samp = [T{h+1}.samp 0];
                else
                    fsample_g = objfun(xxx_g);
                    T{h+1}.samp = [T{h+1}.samp 1];

                    X_sample = [X_sample; xxx_g];
                    F_sample = [F_sample; fsample_g];
                    Nsmp = Nsmp+1;
                    b_hi_max_2 = max(b_hi_max_2, fsample_g);
                    if  result_save == 1
                        LB = max(F_sample);
                        result = [result; Nsmp, Nspl, LB, rho_bar, xi_max, Tdepth, toc];
                    end
                end
                T{h+1}.f = [T{h+1}.f fsample_g];

                newmin = T{h}.x_min(i_max(h),:);
                T{h+1}.x_min = [T{h+1}.x_min; newmin];
                newmax = T{h}.x_max(i_max(h),:);
                newmax(splitd) = (2*T{h}.x_min(i_max(h),splitd)+T{h}.x_max(i_max(h),splitd))/3.0;
                T{h+1}.x_max = [T{h+1}.x_max; newmax];
                T{h+1}.leaf = [T{h+1}.leaf 1];

                % right node
                T{h+1}.x = [T{h+1}.x;x_d];
                xxx_d = x_lower + x_d .* x_width;
                UCB = +inf;
                if GP_use == 1
                    [m, s2] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx_d);
                    UCB = m+(GP_varsigma(Nucb)+0.2)*sqrt(s2);
                end
                if UCB <= LB && GP_use == 1
                    Nucb = Nucb + 1;
                    fsample_d = UCB;
                    T{h+1}.samp = [T{h+1}.samp 0];
                else
                    fsample_d = objfun(xxx_d);
                    T{h+1}.samp = [T{h+1}.samp 1];

                    X_sample = [X_sample; xxx_d];
                    F_sample = [F_sample; fsample_d];
                    Nsmp = Nsmp+1;
                    b_hi_max_2 = max(b_hi_max_2, fsample_d);

                    if  result_save == 1
                        LB = max(F_sample);
                        result = [result; Nsmp, Nspl, LB, rho_bar, xi_max, Tdepth, toc];
                    end
                end
                T{h+1}.f = [T{h+1}.f fsample_d];

                newmax = T{h}.x_max(i_max(h),:);
                T{h+1}.x_max = [T{h+1}.x_max; newmax];
                newmin = T{h}.x_min(i_max(h),:);
                newmin(splitd) = (T{h}.x_min(i_max(h),splitd)+2*T{h}.x_max(i_max(h),splitd))/3.0;
                T{h+1}.x_min = [T{h+1}.x_min; newmin];
                T{h+1}.leaf = [T{h+1}.leaf 1];

                % central node
                T{h+1}.x = [T{h+1}.x;xx];
                T{h+1}.f = [T{h+1}.f T{h}.f(i_max(h))];
                T{h+1}.samp = [T{h+1}.samp 1];
                newmin = T{h}.x_min(i_max(h),:);
                newmax = T{h}.x_max(i_max(h),:);
                newmin(splitd) = (2*T{h}.x_min(i_max(h),splitd)+T{h}.x_max(i_max(h),splitd))/3.0;
                newmax(splitd) = (T{h}.x_min(i_max(h),splitd)+2*T{h}.x_max(i_max(h),splitd))/3.0;
                T{h+1}.x_min = [T{h+1}.x_min; newmin];
                T{h+1}.x_max= [T{h+1}.x_max; newmax];
                T{h+1}.leaf = [T{h+1}.leaf 1];


                % --- output results -------------------------------------------------------
                LB = max(F_sample);

                if result_diplay == 1
                    fprintf(1,'%d, N = %d, n = %d, fmax_hat = %g, rho = %d, xi = %d,h = %d, time = %f \n', Niter, Nsmp, Nspl, LB, rho_bar, xi_max, Tdepth, toc);
                end

            end
        end
        rho_avg = (rho_avg * (Niter - 1) + rho_t) / Niter;
        rho_bar = max(rho_bar,rho_avg);

        %% update Xi
        if LB_old == LB
            XI = max(XI - 2^-1,1);
        else
            XI = XI + 2^2;
        end
        LB_old = LB;

        %% update GP hypper parameters
        if GP_use == 1
            %warning ('off','all');
            if any(Nspl == GP_updates)
                hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, [], X_sample, F_sample);
            end
            %warning ('on','all');
        end


    end

    %% get a maximum point
    f_hi_max = -inf;
    for h=1:h_upper
        if size(T{h}.x,1) < 1, break, end
        for i=1:size(T{h}.x,1)
            f_hi = T{h}.f(i);
            if (f_hi > f_hi_max)
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

    x = x_lower + T{h_max}.x(i_max,:) .* x_width;
    fx = f_hi_max;

end

% GP prediction, make sure std of Gaussian likelihood is large enough
function [m, s2] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx)

    hyp2 = hyp;
    error = 1;
    while error == 1
        error = 0;
        try
           [m, s2] = gp(hyp2, @infExact, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx);
        catch
           error = 1;
           if hyp2.lik == -inf, hyp2.lik = -9; end
           hyp2.lik = hyp2.lik + 1;
        end
    end

end
