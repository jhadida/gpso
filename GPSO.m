classdef GPSO < handle
    
    properties (SetAccess=private)
        srgt % GP surrogate
        tree % sampling tree
    end
    
    properties (Transient,SetAccess=private)
        Niter   % #of iterations
        XI_max  % maximum GP expansion depth
        iterd   % iteration data
    end
    
    properties (Transient,Hidden,SetAccess=private)
        tstart  % initialisation time
        verb    % verbose switch
        obj     % objective function
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
            self.srgt = GP_Surrogate();
            self.tree = GPSO_Tree();
        end
        
        function self=configure( self, sigma, eta )
        %
        % sigma: default 1e-4
        %   Initial log-std of Gaussian likelihood function (normalised units).
        %
        % eta: default 0.05
        %   Probability that UCB < f.
        %
        % JH
            
            if nargin < 2, sigma = 1e-4; end
            if nargin < 3, eta = 0.05; end 
            
            meanfunc = @meanConst; hyp.mean = 0;
            covfunc  = {@covMaterniso, 5}; % isotropic Matern covariance 

            ell = 1/4; 
            sf  = 1;

            % hyper-parameters
            hyp.mean = 0; 
            hyp.lik  = log(sigma); 
            hyp.cov  = log([ell; sf]); 
            
            self.srgt.gpconf( hyp, meanfunc, covfunc, eta );
            
        end
        
        function out = run( self, objfun, domain, Nmax, upc, verb )
        %
        % objfun:
        %   Function handle taking a candidate sample and returning a scalar.
        %   Candidate sample size will be 1 x Ndim.
        %   The optimisation MAXIMISES this function.
        %
        % domain:
        %   Ndim x 2 matrix specifying the boundaries of the hypercube.
        %
        % Nmax:
        %   Maximum number of function evaluation.
        %   This can be considered as a "budget" for the optimisation.
        %   Note that the actual number of evaluations can exceed this value (usually not by much).
        %
        % update: default floor(logspace(1,5))
        %   Update timings for GP hyperparameters, in number of node splits:
        %       - if empty, update after every iteration;
        %       - if any is null or negative, never update;
        %       - otherwise consume vector as the number of splits exceeds the next update.
        %
        % verb: default true
        %   Verbose switch.
        %
        % JH
        
            Ndim = size(domain,1);
            if nargin < 6, verb=true; end
            if nargin < 5, upc=2*Ndim; end
            
            self.verb  = verb;
            self.iterd = {};
            
            % initialisation
            self.info( 'Starting %d-dimensional optimisation, with a budget of %d evaluations...', Ndim, Nmax );
            self.initialise( objfun, domain );
            self.notify( 'PostInitialise' );
            
            % iterate
            LB  = self.srgt.best_score();
            XI  = 1;
            upn = 1;
            upc = upc/2;
            while self.srgt.Ne < Nmax
                
                self.Niter = self.Niter + 1;
                self.info('\n\t------------------------------ Elapsed time: %g sec', toc(self.tstart));
                self.info('\tIteration #%d (depth: %d, neval: %d, score: %g)', ...
                    self.Niter, self.tree.depth, self.srgt.Ne, LB );
                
                % run steps
                LB = self.step_1(LB);
                [i_max,k_max,g_max] = self.step_2();
                [i_max,k_max] = self.step_3(i_max,k_max,g_max,XI);
                
                if any(i_max)
                    self.step_4(i_max,k_max); 
                else
                    warning( 'No remaining leaf after step 3, aborting.' );
                    break;
                end
                
                % update lower bound
                LB_old = LB;
                LB = self.srgt.best_score();
                
                % update iteration data
                self.iterd{end+1} = [XI, nnz(i_max), LB];
                
                % update XI (line 38)
                if LB_old == LB
                    XI = max( 1, XI - 2^-1 );
                else
                    XI = min( self.XI_max, XI + 2^2 );
                end
                
                % update GP hyper parameters
                if self.tree.Ns >= (upc*upn*(upn+1))
                    self.info('\tHyperparameter update (n=%d).',upn);
                    self.srgt.gp_update();
                    upn = upn+1; 
                    self.notify( 'PostUpdate' );
                end
            
                self.notify( 'PostIteration' );
                
            end
            
            self.notify( 'PreFinalise' );
            out = self.finalise();
            
        end
        
    end
    
    methods (Hidden,Access=private)
        
        % print messages
        function info(self,fmt,varargin)
            if self.verb
                fprintf( [fmt '\n'], varargin{:} );
            end
        end
        
        % return when the next hyperparameter update is due
        function next = find_next(self,next,update)
            
            if next == 0 || isinf(next)
                return;
            end
            
            k = find( update > self.tree.Ns, 1, 'first' ); 
            if isempty(k)
                next = Inf;
                warning(sprintf([ ...
                    'Number of splits exceeded largest update timing for GP hyperparameters.\n' ...
                    'This means the GP hyperparameters will not be updated from now on (Nsplits=%d).\n' ...
                    'Consider adding larger update timings in future runs (see input "update" to the run method).' ...
                ],self.tree.Ns)); %#ok
            else
                next = update(k);
            end
            
        end
        
    end
    
    methods (Hidden,Access=private)
        
        function initialise(self,objfun,domain)
            
            if isempty(which('gp'))
                gpml_start();
            end
            
            self.Niter  = 0; 
            self.tstart = tic;
            self.obj    = objfun;
            
            % max exploration depth
            Nd = size(domain,1);
            switch floor(Nd/10)
                case 0 % below 10
                    self.XI_max = 4;
                case 1 % below 20
                    self.XI_max = 3;
                otherwise % 20 and more
                    self.XI_max = 2;
            end
            
            % initialise tree
            self.tree.init(Nd);
            
            % initialise surrogate
            self.srgt.init( domain );
            x_init = mean(domain'); %#ok
            f_init = self.obj(x_init);
            self.srgt.append( x_init, f_init, 0, false );
            
        end
        
        function out = finalise(self)
            
            % list all evaluated samples
            [x,f] = self.srgt.samp_evaluated();
            out.samp.x = x;
            out.samp.f = f;
            
            % get best sample
            [x,f] = self.srgt.best_sample();
            out.sol.x = x;
            out.sol.f = f;
            
            self.info('Best score out of %d samples: %g', numel(out.samp.f), out.sol.f);
            self.info('Total runtime: %f sec',toc( self.tstart ));
            gpml_stop();
            
        end
        
        function LB = step_1(self,LB)
            
            self.info('\tStep 1:');
            
            % update UCB
            self.info('\t\tUpdate UCB.');
            self.srgt.ucb_update();
            
            % find leaves with UCB > LB 
            % NOTES: 
            % 1. we know these are GP-based, if any, because LB is updated before each iteration
            % 2. there are no non-leaf GP-based nodes, because of step 2
            k = find( self.srgt.ucb > LB ); 
            n = numel(k);
            
            % evaluate those samples and update UCB again
            if n > 0
                self.info('\t\tFound %d nodes with UCB > LB, evaluating...',n);
                x = self.srgt.coord( k, true );
                f = nan(n,1);
                for i = 1:n
                    f(i) = self.obj(x(i,:));
                end
                self.srgt.edit( k, f );
                self.srgt.ucb_update();
                LB = max([ LB; f ]);
                self.info('\t\tNew best score is: %g',LB);
            end
            
        end
        
        function [i_max,k_max,g_max] = step_2(self)
            
            self.info('\tStep 2:');
            depth = self.tree.depth;
            i_max = zeros(depth,1);
            k_max = zeros(depth,1);
            g_max = -inf(depth,1);
            upucb = false; 
            v_max = -inf;
            
            for h = 1:depth
                
                v_bak = v_max;
                while true
                    
                    % restore maximum value so far
                    v_max = v_bak; 
                    
                    % find leaf node with score greater than any larger leaf node
                    width = self.tree.width(h);
                    for i = 1:width
                        if self.tree.leaf(h,i)
                            k = self.tree.samp(h,i);
                            g_hi = self.srgt.ucb(k);
                            if g_hi > v_max
                                v_max = g_hi;
                                i_max(h) = i;
                                k_max(h) = k;
                                g_max(h) = g_hi;
                            end
                        end
                    end
                    
                    kmax = k_max(h);
                    if (kmax > 0) && self.srgt.gp_based(kmax)
                        self.info('\t\t[h=%02d] Sampling GP-based leaf %d with UCB %g',h,kmax,v_max);
                        self.srgt.edit( kmax, self.obj(self.srgt.coord(kmax,true)) );
                        upucb = true;
                    else
                        break; % either no selection, or selection is already sampled
                    end
                                        
                end % while
                
                if i_max(h)
                    self.info('\t\t[h=%02d] Select leaf %d with score %g',h,i_max(h),v_max);
                else
                    self.info('\t\t[h=%02d] No leaf selected',h);
                end
                
            end % for
            
            if upucb
                self.info('\t\tUpdating UCB.');
                self.srgt.ucb_update();
            end
            
        end
        
        function [i_max,k_max] = step_3(self,i_max,k_max,g_max,XI)
            
            self.info('\tStep 3:');
            depth = self.tree.depth;
            
            % local assignment for convenience, not to be returned!
            XI = min( ceil(XI), self.XI_max ); 
            
            % number of UCB that would be used if the current number of selected leaves were split
            M = self.srgt.Ng + 2*nnz(i_max);
            
            for h = 1:depth
            if i_max(h) > 0
                
                % Search depth:
                %   - cannot be deeper than the tree (duh), 
                %   - is bounded by XI_max.
                sdepth = 0;
                h2_max = min( depth, h+XI );
                for h2 = (h+1) : h2_max 
                    if i_max(h2) > 0
                        sdepth = h2 - h; break;
                    end
                end
                if sdepth == 0, continue; end
                
                % Find out whether any downstream interval has a UCB greater than 
                % currently best known score at matched depth.
                %
                % Do this by artificially expanding the GP tree and using GP-UCB
                % to compute expected scores.
                T = repstruct( {'lower','upper','coord'}, sdepth+1, 1 );
                
                T(1).lower = self.tree.lower(h,i_max(h));
                T(1).upper = self.tree.upper(h,i_max(h));
                T(1).coord = self.srgt.coord(k_max(h));
                
                z_max = -inf;
                for h2 = 1:sdepth
                    for i2 = 1:3^(h2-1)

                        [g,d,x,s]  = split_largest_dimension( T(h2), i2, T(h2).coord(i2,:) );
                        [mu,sigma] = self.srgt.gp_call( [g;d] );
                        z_max      = max( mu + self.srgt.GP.varsigma(M)*sigma );

                        if z_max >= g_max(h+sdepth), break; end % early cancelling

                        U = split_tree( T(h2), i2, g, d, x, s );
                        T(h2+1).coord = [ T(h2+1).coord; U.coord ];
                        T(h2+1).lower = [ T(h2+1).lower; U.lower ];
                        T(h2+1).upper = [ T(h2+1).upper; U.upper ];

                    end
                    if z_max >= g_max(h+sdepth), break; end % "chain-break"
                end
                
                % If none of the downstream intervals has an "interesting" score, ignore it for this iteration.
                if z_max < g_max(h+sdepth)
                    i_max(h) = 0; 
                    k_max(h) = 0;
                    self.info('\t\t[h=%02d,search=%d] Drop selection (expected=%g < known=%g)',h,sdepth,z_max,g_max(h+sdepth));
                else
                    self.info('\t\t[h=%02d,search=%d] Maintain selection with expected score %g',h,sdepth,z_max);
                end
                
            end % if
            end % for
            
        end
        
        function step_4(self,i_max,k_max)
            
            self.info('\tStep 4:');
            depth = self.tree.depth;
            
            for h = 1:depth
            if i_max(h) > 0
                
                imax = i_max(h);
                kmax = k_max(h);
                
                % Split leaf along largest dimension
                [g,d,x,s] = split_largest_dimension( self.tree.level(h), imax, self.srgt.coord(kmax) );
                
                % Compute extents of new intervals
                U = split_tree( self.tree.level(h), imax, g, d, x, s );
                [mu,sigma] = self.srgt.gp_call( [g;d] );
                k = self.srgt.append( [g;d], mu, sigma, true );
                
                % Commit split to tree member
                self.tree.split( [h,imax], U.lower, U.upper, [k,kmax] );
                self.info('\t\t[h=%02d] Split dimension %d of leaf %d',h,s,imax);
                
            end % if
            end % for
            
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
%       T.lower(k,:)              T.upper(k,:)
% Lvl      \                         /
% k:        =-----------x-----------=
% 
%
% k+1:      =---g---=---x---=---d---=
%          /        |       |        \
%        Tmin     Gmax     Dmin      Tmax
%

function [g,d,x,s] = split_largest_dimension(T,k,x)

    g = x;
    d = x;

    Tmin = T.lower(k,:);
    Tmax = T.upper(k,:);
    
    [~,s] = max( Tmax - Tmin );
    g(s)  = (5*Tmin(s) +   Tmax(s))/6;
    d(s)  = (  Tmin(s) + 5*Tmax(s))/6;
    
end

function U = split_tree(T,k,g,d,x,s)

    Tmin = T.lower(k,:);
    Tmax = T.upper(k,:);
    
    Gmax = Tmax;
    Dmin = Tmin;
    Xmin = Tmin;
    Xmax = Tmax;
    
    Gmax(s) = (2*Tmin(s) +   Tmax(s))/3.0;
    Dmin(s) = (  Tmin(s) + 2*Tmax(s))/3.0;
    Xmin(s) = Gmax(s);
    Xmax(s) = Dmin(s);
    
    U.coord = [g;d;x];
    U.lower = [Tmin;Dmin;Xmin];
    U.upper = [Gmax;Tmax;Xmax];

end
