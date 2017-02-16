classdef GPSO < handle
    
    properties (SetAccess=private)
        tree
        samp
        
        GP
        GP_use
        GP_norm
    end
    
    properties (Hidden,Transient,SetAccess=private)
        Nsamp   % #of objective evaluations
        Niter   % #of iterations
        Nsplit  % #of leaves split
        Nmax    % maximum #of objective evaluations
        Ndim    % #of dimensions
        Nucb    % #of times UCB was used instead of sample
        LB      % best score so far
        depth   % depth of the tree
        XI_max  % maximum GP expansion depth
        tstart  % initialisation time
        verb    % verbose switch
        dev     % development options
    end
    
    events
        PostInitialise
        PostIteration
        PostUpdate
        PreFinalise
    end
    
    methods
        
        function self = GPSO()
            self.clear();
        end
        
        function self=clear(self)
            
            self.tree    = [];
            self.samp    = struct();
            self.GP      = struct();
            self.GP_use  = false;
            self.GP_norm = 1;
            
            self.dev.gmax = false;
            self.dev.ucb  = false;
            self.dev.gpu  = false;
            self.dev.lb   = false;
            
        end
        
        % set/unset GP
        function self=set_GP( self, hyp, likfunc, meanfunc, covfunc, update, eta )
            
            if nargin < 7, eta=0.05; end
            if nargin < 6, update=[1:6,8,floor(logspace(1,5))]; end
            
            self.GP_use = true;
            self.GP.hyp = hyp;
            self.GP.likfunc = likfunc;
            self.GP.meanfunc = meanfunc;
            self.GP.covfunc = covfunc;
            self.GP.varsigma = @(M) sqrt(4*log(pi*M) - 2*log(12*eta));
            self.GP.update = update;
            
        end
        function self=unset_GP(self)
            self.GP_use = false;
            self.GP = struct();
        end
        
        % default options
        function self=set_defaults( self, sigma, norm )
            
            if nargin < 2, sigma = 1e-4; end
            if nargin < 3, norm = 1; end
            
            likfunc  = @likGauss;  % Gaussian likelihood
            meanfunc = @meanConst; %  Constant mean 
            covfunc  = {@covMaterniso, 5}; % isotropic Matern covariance 

            ell = 1/4; sf = 1; 

            hyp.mean = 0; 
            hyp.lik  = log(sigma); 
            hyp.cov  = log([ell; sf]); % hyper-parameters
            
            self.GP_norm = norm;
            self.set_GP( hyp, likfunc, meanfunc, covfunc );
            
        end
        
        % run optimisation
        function out = run( self, objfun, domain, neval, verb )

            if nargin < 5, verb=true; end
            self.verb = verb;
            
            % initialisation
            self.LB = self.initialise( objfun, domain, neval );
            self.notify( 'PostInitialise' );
            
            % iterate
            next = 1;
            rho = [];
            XI = 1;
            
            while self.Nsamp < self.Nmax
                
                % update lower bound
                LB_old = self.LB;
                self.inc_iter();
                
                % print information
                self.info('\n\t------------------------------');
                self.info('\tIteration #%d (depth: %d, nsample: %d, score: %g, time: %g sec)', ...
                    self.Niter, self.depth, self.Nsamp, self.LB, toc(self.tstart) );
                
                % run steps
                [i_max,g_max] = self.step_1(objfun);
                if self.GP_use
                    i_max = self.step_2(i_max,g_max,XI);
                end
                rho(self.Niter) = self.step_3(objfun,i_max,g_max); %#ok
                
                % update XI
                self.LB = self.max_samp();
                if LB_old == self.LB
                    XI = max( 1, XI-2^-1 );
                else
                    XI = XI + 2^2;
                end
                
                % update GP hyper parameters
                if self.GP_use 
                    next = self.gp_update(next);
                end
                self.notify( 'PostIteration' );
                
            end
            
            self.notify( 'PreFinalise' );
            out = self.finalise();
            
        end
        
        % normalise/denormalise candidate samples
        function y = normalise(self,x)
            y = bsxfun( @minus, x, self.samp.lower );
            y = bsxfun( @rdivide, y, self.samp.delta );
        end
        function y = denormalise(self,x)
            y = bsxfun( @times, x, self.samp.delta );
            y = bsxfun( @plus, y, self.samp.lower );
        end
        
        % maximum score obtained across all samples
        function f = max_samp(self)
            n = self.Nsamp;
            f = max(self.samp.f(1:n));
        end
        
        % get copy of current samples
        function [x,f] = get_samples(self,denorm)
            if nargin < 2, denorm=true; end
            
            n = self.Nsamp;
            x = self.samp.x(1:n,:); 
            f = self.samp.f(1:n);
            
            if denorm
                x = self.denormalise(x);
            end
        end
        
        % get output
        function out = get_output(self)
            
            % get samples
            n = self.Nsamp;
            out.samp.x = self.denormalise(self.samp.x(1:n,:));
            out.samp.f = self.samp.f(1:n);
            
            % find maximum point
            [f,k] = max(out.samp.f);
            
            out.sol.f = f;
            out.sol.x = out.samp.x(k,:);
            
        end
        
        % call the GPML library for prediction at input coordinates x
        function [mu,sigma] = gp_call( self, x )
            
            assert( ~isempty(which('gp')), 'The GPML library is not on the path (use gpml_start).' );
            
            hyp = self.GP.hyp;
            err = true;
            
            x = self.gp_prep(x);
            [xsample,fsample] = self.gp_data();
            
            while err
                err = false;
                try
                    [mu,sigma] = gp( hyp, @infExact, ...
                        self.GP.meanfunc, self.GP.covfunc, self.GP.likfunc, ...
                        xsample, fsample, x ...
                    );
                catch
                    err = true;
                    if hyp.lik == -inf, hyp.lik = -9; end
                    hyp.lik = hyp.lik + 1;
                end
            end
            if self.dev.gpu
                self.GP.hyp = hyp; % JH: commit?
            end
            
            % gp returns the variance, not the std
            sigma = sqrt(sigma);
            
        end
        
    end
    
    methods (Hidden,Access=private)
        
        % increment counters
        function n=inc_samp(self)
            n = self.Nsamp;
            n = n+1;
            self.Nsamp = n;
        end
        
        function n=inc_split(self)
            n = self.Nsplit;
            n = n+1;
            self.Nsplit = n;
        end
        
        function n=inc_iter(self)
            n = self.Niter;
            n = n+1;
            self.Niter = n;
        end
        
        function n=inc_ucb(self)
            n = self.Nucb;
            n = n+1;
            self.Nucb = n;
        end
        function n=dec_ucb(self)
            n = self.Nucb;
            n = max(0,n-1);
            self.Nucb = n;
        end
        
        % for progress messages
        function info(self,fmt,varargin)
            if self.verb
                fprintf( [fmt '\n'], varargin{:} );
            end
        end
        
        % number of samples at level h
        function n=tree_nx(self,h)
            n = size(self.tree(h).x,1);
        end
        
        % save candidate sample
        function save_sample(self,x,f)
            
            n = self.inc_samp();
            if self.dev.lb
                self.LB = max( self.LB, f ); % JH: may be better to do it at each iteration instead
            end
            
            self.samp.x(n,:) = x;
            self.samp.f(n)   = f;
            
        end
        
        % get GP-friendly samples
        function x = gp_prep(self,x)
            if self.GP_norm > 0
                x = self.GP_norm * x;
            else
                x = self.denormalise(x);
            end
        end
        function [x,f] = gp_data(self)
            n = self.Nsamp;
            x = self.gp_prep(self.samp.x(1:n,:));
            f = self.samp.f(1:n);
        end
        
        % update GP hyperparameters
        function next = gp_update(self,next)
            
            if self.Nsplit >= self.GP.update(next)
                    
                % careful with next update: we might have skipped several ones in last iteration
                next = find( self.GP.update > self.Nsplit, 1, 'first' ); 
                self.info('\tHyperparameter update (Nsplit=%d, next=%d).',self.Nsplit,self.GP.update(next));

                [xsample,fsample] = self.gp_data();
                self.GP.hyp = minimize( self.GP.hyp, @gp, -100, ...
                    @infExact, self.GP.meanfunc, self.GP.covfunc, self.GP.likfunc, xsample, fsample );

                % JH: don't allow sigma to become too small
                self.GP.hyp.lik = max( self.GP.hyp.lik, -15 );

                if isempty(next)
                    warning(sprintf([ ...
                        'Number of splits exceeded largest update timing for GP hyperparameters.\n' ...
                        'This means the GP hyperparameters will not be updated from now on (Nsplits=%d).\n' ...
                        'Consider adding larger update timings in future runs (see set_defaults and set_GP).' ...
                    ],self.Nsplit)); %#ok
                end
                self.notify( 'PostUpdate' );

            end
            
        end
        
    end
    
    methods (Hidden,Access=private)
        
        function f_init = initialise(self,objfun,domain,neval)
            
            assert( size(domain,2)==2, 'Domain should be Nx2.' );
            assert( size(domain,1)>0, 'Number of dimensions should be positive.' );
            assert( neval > 0, 'Max number of evaluations should be >0.' );
            
            if isempty(which('gp'))
                gpml_start();
            end
            
            h_pre = max( 100, sqrt(neval) ); % used for preallocation, but non-limitting (see footnote p.7)
            ndim  = size(domain,1);
            self.info( 'Starting %d-dimensional optimisation, with a budget of %d evaluations...', ndim, neval );
            
            % dimensions & counters
            self.Nsamp  = 0; % #of times the objective has been evaluated
            self.Niter  = 0; % #of iterations in the main loop (each loop can evaluate multiple times)
            self.Nsplit = 1; % #of interval splitting (step 4)
            self.Nmax   = neval; % maximum #of objective evaluations
            self.Ndim   = ndim; % dimensionality of search space
            self.Nucb   = 1;
            self.LB     = -inf;
            self.depth  = 1;
            self.tstart = tic;
            
            % XI_max
            switch floor(ndim/10)
                case 0
                    self.XI_max = 4;
                case 1
                    self.XI_max = 3;
                otherwise
                    self.XI_max = 2;
            end
            
            % initialise sampling
            x_lower = domain(:,1)';
            x_upper = domain(:,2)';
            x_delta = x_upper - x_lower;
            x_init  = (x_upper + x_lower)/2;
            f_init  = objfun(x_init);

            assert( all(x_delta > eps), 'Domain is too narrow.' );
            
            self.samp.lower = x_lower;
            self.samp.upper = x_upper;
            self.samp.delta = x_delta;

            self.samp.x = nan( neval, ndim );
            self.samp.f = nan( neval, 1 );
            
            x_init = self.normalise(x_init);
            self.save_sample( x_init, f_init );

            % initialise tree
            self.tree = repstruct( {'x_max','x_min','x','f','leaf','samp'}, [h_pre,1] );
            
            T = self.tree(1);
            T.x_min = zeros(1,ndim);
            T.x_max = ones(1,ndim);
            T.x = x_init;
            T.f = f_init;
            T.leaf = 1;
            T.samp = 1;
            self.tree(1) = T;
            
        end
        
        function out = finalise(self)
            
            n = self.Nsamp;
            d = self.depth;
            
            % remove unused allocation
            self.tree   = self.tree(1:d);
            self.samp.x = self.samp.x(1:n,:);
            self.samp.f = self.samp.f(1:n);
            
            % get output
            out = self.get_output();
            
            % information
            self.info('Best score out of %d samples: %g', self.Nsamp, out.sol.f);
            self.info('Total runtime: %s',dk.time.sec2str(toc( self.tstart )));
            gpml_stop();
            
        end
        
        function [i_max,g_max] = step_1(self,objfun)
            
            self.info('\tStep 1:');
            i_max = zeros(self.depth,1);
            g_max = -inf(self.depth,1);
            
            v_max = -inf;
            for h = 1:self.depth
                
                stop  = false;
                v_bak = v_max;
                while ~stop
                    
                    % JH: restore maximum value 
                    v_max = v_bak;
                    
                    % Find leaf interval with score greater than any larger leaf interval.
                    nx = self.tree_nx(h);
                    for i = 1:nx
                        if self.tree(h).leaf(i) == 1
                            g_hi = self.tree(h).f(i);
                            if g_hi > v_max
                                v_max = g_hi;
                                i_max(h) = i;
                                g_max(h) = g_hi;
                            end
                        end
                    end
                    
                    imax = i_max(h);
                    if imax == 0, break, end
                    
                    % If selected value is GP-based, then sample it and restart selection.
                    if self.tree(h).samp(imax) == 1
                        stop = true;
                    else
                        xsample = self.tree(h).x(imax,:);
                        fsample = objfun(self.denormalise(xsample));
                        
                        self.info('\t\t[h=%02d] Sampling GP-based leaf %d with score %g',h,imax,v_max);
                        self.save_sample( xsample, fsample );
                        self.dec_ucb(); % JH: algorithm line 17
                        self.tree(h).samp(imax) = 1;
                        
                        % JH:
                        % No need to update g_max(h) after sampling because 
                        % we restart the selection.
                    end
                    
                end % while
                if imax
                    self.info('\t\t[h=%02d] Select leaf %d with score %g',h,imax,v_max);
                else
                    self.info('\t\t[h=%02d] No leaf selected',h);
                end
                
            end % for
            
        end
        
        function i_max = step_2(self,i_max,g_max,XI)
            
            self.info('\tStep 2:');
            for h = 1:self.depth    
            if i_max(h) > 0
                
                % Search depth:
                %   - cannot be deeper than the tree (duh), 
                %   - is bounded by XI_max.
                ki = 0;
                h2_max = min( self.depth, h+min(ceil(XI),self.XI_max) );
                for h2 = (h+1) : h2_max 
                    if i_max(h2) > 0
                        ki = h2 - h; break;
                    end
                end
                if ki == 0, continue; end
                
                % Find out whether any downstream interval has a UCB greater than 
                % currently best known score at matched depth.
                %
                % Do this by artificially expanding the GP tree and using GP-UCB
                % to compute expected scores.
                T = repstruct( {'x_max','x_min','x'}, ki+1, 1 );
                M = self.Nucb;
                
                T(1).x_max = self.tree(h).x_max(i_max(h),:);
                T(1).x_min = self.tree(h).x_min(i_max(h),:);
                T(1).x     = self.tree(h).x(i_max(h),:);
                
                z_max = -inf;
                for h2 = 1:ki
                    for i2 = 1:3^(h2-1)

                        [g,d,x,s]  = split_largest_dimension( T(h2), i2 );
                        [mu,sigma] = self.gp_call( [g;d] ); M = M+2;
                        z_max      = max( mu + self.GP.varsigma(M)*sigma );

                        if z_max >= g_max(h+ki), break; end

                        U = split_tree( T(h2), i2, g, d, x, s );
                        T(h2+1).x     = [ T(h2+1).x;     U.x     ];
                        T(h2+1).x_min = [ T(h2+1).x_min; U.x_min ];
                        T(h2+1).x_max = [ T(h2+1).x_max; U.x_max ];

                    end
                    if z_max >= g_max(h+ki), break; end % "chain-break"
                end
                
                % If none of the downstream intervals has an "interesting" score, ignore it for this iteration.
                if z_max < g_max(h+ki)
                    self.Nucb = M; % if we actually used M UCBs, we update Nucb = M;
                    i_max(h)  = 0; % if it turns out that some UCB exceeded g_max(h+ki), then it does not matter 
                                   % whether or not f <= UCB. It may be UCB < f, and still it works exactly same. 
                                   % So, we do not have to update Nucb in this case.
                    self.info('\t\t[h=%02d,ki=%d] Drop selection (expected=%g < known=%g)',h,ki,z_max,g_max(h+ki));
                else
                    self.info('\t\t[h=%02d,ki=%d] Maintain selection with expected score %g',h,ki,z_max);
                end
                
            end % if
            end % for
            
        end
        
        function rho = step_3(self,objfun,i_max,g_max)
            
            v_max = -inf;
            rho = 0;
            
            self.info('\tStep 3:');
            for h = 1:self.depth
            if i_max(h) > 0
                
                if g_max(h) < v_max % JH: >= instead of >
                    self.info('\t\t[h=%02d] Drop selection (known=%g < upstream=%g)',h,g_max(h),v_max);
                    continue;
                elseif self.dev.gmax
                    v_max = g_max(h); % JH: should we do that?
                end
                
                rho = rho+1;
                self.depth = max( self.depth, h+1 );
                
                % Split the leaf along largest dimension
                self.tree(h).leaf(i_max(h)) = 0;
                [g,d,x,s] = split_largest_dimension( self.tree(h), i_max(h) );
                self.inc_split();
                
                % Compute extents of new intervals
                U  = split_tree( self.tree(h), i_max(h), g, d, x, s );
                xf = self.tree(h).f(i_max(h));
                xs = self.tree(h).samp(i_max(h)); % JH: only valid if x is unchanged (cf split_largest_dimension)!
                % assert( xs == 1 ); We know it is 1 because of step_1
                
                % Append new intervals with extents
                self.tree(h+1).x     = [ self.tree(h+1).x;     U.x     ];
                self.tree(h+1).x_min = [ self.tree(h+1).x_min; U.x_min ];
                self.tree(h+1).x_max = [ self.tree(h+1).x_max; U.x_max ];
                
                % Compute expected scores in new intervals
                [gf,gs,v_max] = self.subroutine_UCB( objfun, g, v_max );
                [df,ds,v_max] = self.subroutine_UCB( objfun, d, v_max );
                
                self.tree(h+1).f    = [self.tree(h+1).f,    gf,df,xf];
                self.tree(h+1).samp = [self.tree(h+1).samp, gs,ds,xs];
                self.tree(h+1).leaf = [self.tree(h+1).leaf,  1, 1, 1];
                
                self.info('\t\t[h=%02d] Split dim %d, %d/3 sampled, old=%g, new=%g',h,s,gs+ds+xs,g_max(h),v_max);
                
            end % if
            end % for
            
        end
        
        function [f,s,v_max] = subroutine_UCB(self,objfun,x,v_max)
            
            % UCB at point x (force sampling if not using GP)
            if self.GP_use
                [mu,sigma] = self.gp_call(x);
                if self.dev.ucb
                    UCB = mu + (self.GP.varsigma(self.Nucb)+0.2)*sigma; % JH: UCB boosting??
                else
                    UCB = mu + self.GP.varsigma(self.Nucb)*sigma; 
                end
            else
                UCB = +inf;
            end 
            
            % Sample only if UCB exceeds overall best known score
            if UCB <= self.LB
                % Need to update Nucb only if we require f <= UCB. 
                % The other case f > UCB does not require to take union bound.
                self.inc_ucb();
                f = UCB;
                s = 0;
            else
                f = objfun(self.denormalise(x));
                s = 1;
                
                self.save_sample( x, f );
                v_max = max( v_max, f );
            end
            
        end
        
    end
    
end

% create struct-array of required size
function s = repstruct( fields, varargin )

    n = numel(fields);
    s = cell(1,2*n);
    s(1:2:end) = fields;
    s = struct(s{:});
    s = repmat( s, varargin{:} );

end

% 
%         T.x_min(k,:)              T.x_max(k,:)
% Lvl      \                         /
% k:        =-----------x-----------=
% 
%
% k+1:      =---g---=---x---=---d---=
%          /        |       |        \
%        Tmin     Gmax     Dmin      Tmax
%

function [g,d,x,s] = split_largest_dimension(T,k)

    x = T.x(k,:);
    g = x;
    d = x;

    Tmin = T.x_min(k,:);
    Tmax = T.x_max(k,:);
    
    [~,s] = max( Tmax - Tmin );
    g(s)  = (5*Tmin(s) +   Tmax(s))/6;
    d(s)  = (  Tmin(s) + 5*Tmax(s))/6;
    
    % Not necessary if initial point is at the center.
    % If uncommented, indicate that x has NOT been sampled (cf step 4-5)!
    %x(s)  = (  Tmin(s) +   Tmax(s))/2; 

end

function U = split_tree(T,k,g,d,x,s)

    Tmin = T.x_min(k,:);
    Tmax = T.x_max(k,:);
    
    Gmax = Tmax;
    Dmin = Tmin;
    Xmin = Tmin;
    Xmax = Tmax;
    
    Gmax(s) = (2*Tmin(s) +   Tmax(s))/3.0;
    Dmin(s) = (  Tmin(s) + 2*Tmax(s))/3.0;
    Xmin(s) = Gmax(s);
    Xmax(s) = Dmin(s);
    
    U.x     = [g;d;x];
    U.x_min = [Tmin;Dmin;Xmin];
    U.x_max = [Gmax;Tmax;Xmax];

end
