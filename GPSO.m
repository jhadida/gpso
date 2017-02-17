classdef GPSO < handle
    
    properties (SetAccess=private)
        srgt % GP surrogate
        tree % sampling tree
        iter % iteration data
        verb % verbose switch
    end
    
    properties (Transient,Dependent)
        Niter;
    end
    
    events
        PostInitialise
        PostIteration
        PostUpdate
        PreFinalise
    end
    
    
    
    
    %% EXPOSED INTERFACE
    %
    % All the functions that a typical user will call.
    %
    % Housekeeping (constructor + cleanup)
    % Runtime (configuration + run)
    % Post-analysis (exploration + serialisation).
    %
    methods
        
        function self = GPSO()
            self.clear();
            self.configure(); % set defaults
        end
        
        function self=clear(self)
            self.srgt = GP_Surrogate();
            self.tree = GPSO_Tree();
            self.iter = {};
            self.verb = true;
        end
        
        % dependent parameter to get the current iteration count
        % useful when working with event callbacks
        function n = get.Niter(self), n=1+numel(self.iter); end
        
        function self=configure( self, sigma, varsigma )
        %
        % sigma: default 1e-4
        %   Initial log-std of Gaussian likelihood function (normalised units).
        %
        % varsigma: default 3
        %   Controls the probability that UCB < f.
        %
        % JH
            
            if nargin < 2, sigma = 1e-3; end
            if nargin < 3, varsigma = erfcinv(0.005); end 
            
            meanfunc = @meanConst; hyp.mean = 0;
            covfunc  = {@covMaterniso, 5}; % isotropic Matern covariance 

            ell = 1/4; 
            sf  = 1;

            % hyper-parameters
            hyp.mean = 0; 
            hyp.lik  = log(sigma); 
            hyp.cov  = log([ell; sf]); 
            
            self.srgt.set_gp( hyp, meanfunc, covfunc );
            self.srgt.set_varsigma_const( varsigma );

        end
        
        function out = run( self, objfun, domain, Neval, Nsamp, upc, verb )
        %
        % objfun:
        %   Function handle taking a candidate sample and returning a scalar.
        %   Candidate sample size will be 1 x Ndim.
        %   The optimisation MAXIMISES this function.
        %
        % domain:
        %   Ndim x 2 matrix specifying the boundaries of the hypercube.
        %
        % Neval:
        %   Maximum number of function evaluation.
        %   This can be considered as a "budget" for the optimisation.
        %   Note that the actual number of evaluations can exceed this value (usually not by much).
        %
        % upc: default 2*Ndims
        %   Update constant for GP hyperparameters.
        %   See function update_gp for currently selected method.
        %
        % verb: default true
        %   Verbose switch.
        %
        % JH
        
            assert( ismatrix(domain) && ~isempty(domain) && ... 
                size(domain,2)==2 && all(diff(domain,1,2) > eps), 'Bad domain.' );
        
            Ndim = size(domain,1);
            if nargin < 7, verb=true; end
            if nargin < 6 || isempty(upc), upc=1; end
            if nargin < 5 || isempty(Nsamp), Nsamp=3*Ndim^2; end
            
            dk.assert( dk.is.integer(Neval) && Neval>2*Ndim, 'Neval should be >%d.', 2*Ndim );
            dk.assert( dk.is.number(upc) && upc>0, 'upc should be >0.' );
            dk.assert( isscalar(verb) && islogical(verb), 'verb should be boolean.' );
            
            tstart = tic;
            self.iter = {};
            self.verb = verb;
            
            % initialisation
            self.info( 'Starting %d-dimensional optimisation, with a budget of %d evaluations...', Ndim, Neval );
            [i_max,k_max] = self.initialise( domain, objfun );
            self.notify( 'PostInitialise' );
            
            % iterate
            best = self.srgt.best_score();
            upn = self.srgt.Ne;
            
            gpml_start();
            while self.srgt.Ne < Neval
                
                self.info('\n\t------------------------------ Elapsed time: %s', dk.time.sec2str(toc(tstart)) );
                self.info('\tIteration #%d (depth: %d, neval: %d, score: %g)', ...
                    self.Niter, self.tree.depth, self.srgt.Ne, best );
                
                upn = self.step_update(upc,upn);
                self.step_explore(i_max,k_max,Nsamp);
                [i_max,k_max] = self.step_select(objfun);
                
                Nselect = nnz(i_max);
                best = self.srgt.best_score();
                
                if Nselect == 0
                    warning( 'No remaining leaf after step 3, aborting.' );
                    break;
                end
                
                % update iteration data
                self.iter{end+1} = [Nselect, best];
                self.notify( 'PostIteration' );
                
            end
            gpml_stop();
            
            self.notify( 'PreFinalise' );
            out = self.finalise();
            
            self.info('Best score out of %d samples: %g', numel(out.samp.f), out.sol.f);
            self.info('Total runtime: %s', dk.time.sec2str(toc(tstart)) );
            
        end
        
        function node = get_node(self,h,i)
        %
        % h: depth
        % i: depth-specific node index
        %
        % Get a node of the tree, with coordinates and sample.
        %
        
            node = self.tree.node(h,i);
            node.coord = self.srgt.x(node.samp,:);
            node.samp  = self.srgt.y(node.samp,:);
        end
        
        function [best,T] = explore(self,node,depth,varsigma,until)
        %
        % node: either a 1x2 array [h,i] (cf get_node), or a node structure
        % depth: how deep the exploration tree should be 
        %   (WARNING: tree grows exponentially, there will be 3^depth node)
        % varsigma: optimism constant to be used locally for UCB
        % until: early cancelling criterion (stop if one of the samples has a better score)
        %   (NOTE: default is Inf, so full exploration)
        %
        % Explore node by growing exhaustive partition tree, using surrogate for evaluation.
        %
        
            if nargin < 5, until=inf; end
            
            % get node if index were passed
            if ~isstruct(node)
                node = self.get_node(node(1),node(2));
            end
            
            % Temporary exploration tree
            T = dk.struct.repeat( {'lower','upper','coord','samp'}, depth+1, 1 );
            
            T(1).lower = node.lower;
            T(1).upper = node.upper;
            T(1).coord = node.coord;
            T(1).samp  = node.samp;
            
            best = T(1).samp;
            if best(3) >= until, return; end
            
            for h = 1:depth
                for i = 1:3^(h-1)

                    % evaluate GP
                    [g,d,x,s]  = split_largest_dimension( T(h), i, T(h).coord(i,:) );
                    [mu,sigma] = self.srgt.gp_call( [g;d] );

                    % update best score
                    ucb   = mu + varsigma*sigma;
                    [u,k] = max(ucb);
                    if u > best(3)
                        best = [ mu(k), sigma(k), u ];
                    end
                    
                    if u >= until; break; end % early cancelling

                    % record new nodes
                    U = split_tree( T(h), i, g, d, x, s );
                    T(h+1).coord = [ T(h+1).coord; U.coord ];
                    T(h+1).lower = [ T(h+1).lower; U.lower ];
                    T(h+1).upper = [ T(h+1).upper; U.upper ];
                    T(h+1).samp  = [ T(h+1).samp; [mu,sigma,ucb] ];

                end
                if u >= until; break; end % chain-break
            end
            
        end
        
        function [best,S] = explore_samp(self,node,ns,varsigma)
        %
        % node: either a 1x2 array [h,i] (cf get_node), or a node structure
        % ns: number of random points to draw within the node
        % varsigma: optimism constant to be used locally for UCB
        %
        % Explore node by drawing uniformly distributed sample, using surrogate for evaluation. 
        %
            
            % get node if index were passed
            if ~isstruct(node)
                node = self.get_node(node(1),node(2));
            end
            
            % sample points at random in the node
            nd = self.srgt.Nd;
            delta = node.upper - node.lower;
            S.coord = bsxfun( @times, rand(ns,nd), delta );
            S.coord = bsxfun( @plus, S.coord, node.lower );
            
            % evaluate those points
            [mu,sigma] = self.srgt.gp_call( S.coord );
            ucb = mu + varsigma*sigma;
            
            [u,k]  = max(ucb);
            best   = [ mu(k), sigma(k), u ];
            S.samp = [ mu, sigma, ucb ];
            
        end
        
        % serialise data to be saved
        function D = serialise(self,filename)
            D.iter = self.iter;
            D.tree = self.tree.serialise();
            D.surrogate = self.srgt.serialise();
            D.version = '0.1';
            
            if nargin > 1
                save( filename, '-v7', '-struct', 'D' );
            end
        end
        function self=unserialise(self,D)
            
            if ischar(D)
                D = load(D);
            end
            self.iter = D.iter;
            self.tree = GPSO_Tree().unserialise(D.tree);
            self.srgt = GP_Surrogate().unserialise(D.surrogate);
        end
        
    end
    
    
    
    
    %% UTILITIES
    %
    % Functions used internally by the algorithm.
    %
    methods (Hidden,Access=private)
        
        % keeping tabs on number of evaluated samples ..
        function upn=update_samp_linear(self,upc,upn)
            Ne = self.srgt.Ne;
            if (Ne-upn) >= upc

                self.info('\tHyperparameter update (neval=%d).',upn);
                self.srgt.gp_update();
                upn = Ne;
                self.notify( 'PostUpdate' );

            end
        end
        
        % .. or on number of node splits
        function upn=update_split_linear(self,upc,upn)
            Nsplit = self.tree.Ns;
            if Nsplit >= upc*upn

                self.info('\tHyperparameter update (nsplit=%d).',upn);
                self.srgt.gp_update();
                upn = dk.math.nextint( Nsplit/upc );
                self.notify( 'PostUpdate' );

            end
        end
        
        function upn=update_split_quadratic(self,upc,upn)
            Nsplit = self.tree.Ns;
            if 2*Nsplit >= upc*upn*(upn+1)

                self.info('\tHyperparameter update (nsplit=%d).',upn);
                self.srgt.gp_update();
                upn = dk.math.nextint( (sqrt(1+8*Nsplit/upc)-1)/2 );
                self.notify( 'PostUpdate' );

            end
        end
        
        % print formatted messages
        function info(self,fmt,varargin)
            if self.verb
                fprintf( [fmt '\n'], varargin{:} );
            end
        end
        
    end
    
    
    
    
        
    %% RUNTIME BREAKDOWN
    %
    %
    methods (Hidden,Access=private)
        
        function [i_max,k_max] = initialise(self,domain,objfun)
            
            % initialise surrogate
            self.srgt.init( domain );
            nd = self.srgt.Nd;
            nx = 2*nd+1;
            
            % vertices of L1 ball of radius 0.25
            Xinit = 0.5 + [ ...
                -0.25*eye(nd); ...
                +0.25*eye(nd); ...
                zeros(1,nd) ... % centre of the domain
            ]; 
            Finit = nan(nx,1);
            for k = 1:nx
                Finit(k) = objfun(self.srgt.denormalise(Xinit(k,:)));
            end
            Yinit = [Finit, zeros(nx,1), Finit];
            self.srgt.append( Xinit, Yinit );
            
            % initialise tree
            self.tree.init(nd,nx);
            
            % select root
            i_max = 1;
            k_max = nx;
            
        end
        
        function out = finalise(self)
            
            % list all evaluated samples
            [x,f] = self.srgt.samp_evaluated(true);
            out.samp.x = x;
            out.samp.f = f;
            
            % get best sample
            [x,f] = self.srgt.best_sample(true);
            out.sol.x = x;
            out.sol.f = f;
            
        end
        
        % facade method for hyperparameter update
        function upn = step_update(self,upc,upn)
            upn = self.update_samp_linear(upc,upn);
        end
        
        % exploration step: split and sample
        function step_explore(self,i_max,k_max,Nsamp)
            
            self.info('\tStep 1:');
            depth = self.tree.depth;
            varsigma = self.srgt.get_varsigma();
            
            for h = 1:depth
            if i_max(h) > 0
                
                imax = i_max(h);
                kmax = k_max(h);
                
                % Split leaf along largest dimension
                [g,d,x,s] = split_largest_dimension( self.tree.level(h), imax, self.srgt.coord(kmax) );
                U = split_tree( self.tree.level(h), imax, g, d, x, s );
                Uget = @(n) struct( 'lower', U.lower(n,:), 'upper', U.upper(n,:) );
                
                % Explore each new leaf with a uniform sample
                best_g = self.explore_samp( Uget(1), Nsamp, varsigma );
                best_d = self.explore_samp( Uget(2), Nsamp, varsigma );
                best_x = self.explore_samp( Uget(3), Nsamp, varsigma );
                edit_x = [ self.srgt.mu(kmax), 0, best_x(3) ]; % boosting
                
                % Append points and update tree
                k = self.srgt.append( [g;d], [best_g;best_d], true );
                self.srgt.update( kmax, edit_x ); 
                self.tree.split( [h,imax], U.lower, U.upper, [k,kmax] );
                self.info('\t\t[h=%02d] Split dimension %d of leaf %d',h,s,imax);
                
            end % if
            end % for
            
        end
        
        % selection step: select and evaluate
        function [i_max,k_max] = step_select(self,objfun)
            
            self.info('\tStep 2:');
            depth = self.tree.depth;
            i_max = zeros(depth,1);
            k_max = zeros(depth,1);
            v_max = -inf;
            
            for h = 1:depth

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
                        end
                    end
                end

                kmax = k_max(h);
                if (kmax > 0) && self.srgt.is_gp_based(kmax)
                    self.info('\t\t[h=%02d] Sampling GP-based leaf %d with UCB %g',h,kmax,v_max);
                    f = objfun(self.srgt.coord(kmax,true));
                    self.srgt.update( kmax, [f,0,f] ); % Note: important NOT to keep v_max here
                end

                if i_max(h)
                    self.info('\t\t[h=%02d] Select leaf %d with score %g',h,i_max(h),v_max);
                else
                    self.info('\t\t[h=%02d] No leaf selected',h);
                end
                
            end % for
            
        end
        
    end
    
end

% 
%       T.lower(k,:)              T.upper(k,:)
% Lvl      \                         /
% k:        =-----------x-----------=
% 
%
% k+1:      =---g---=---x---=---d---=
%          /        |       |        \
%        Tmin     Gmax     Dmin     Tmax
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
