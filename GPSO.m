classdef GPSO < handle
    
    properties
        verb % verbose switch
        xmet % exploration method
    end
    
    properties (SetAccess=private)
        srgt % GP surrogate
        tree % sampling tree
        iter % iteration data
    end
    
    properties (Transient,Dependent)
        Niter;
    end
    
    % use events to hook additional processing steps
    events
        PostInitialise
        PreIteration
        PostIteration
        PostUpdate
        PreFinalise
    end
    
    
    
    
    %% EXPOSED INTERFACE
    %
    % All the functions that a typical user will call.
    %
    % Housekeeping (constructor + cleanup)
    % Runtime (configuration, run, exploration)
    % I/O (serialisation).
    %
    methods
        
        function self = GPSO(varargin)
            self.clear();
            self.configure(varargin{:}); 
        end
        
        function self=clear(self)
            self.srgt = GP_Surrogate();
            self.tree = GPSO_Tree();
            self.iter = {};
            self.verb = true;
            self.xmet = 'tree';
        end
        
        % dependent parameter to get the current iteration count
        % useful when working with event callbacks
        function n = get.Niter(self), n=numel(self.iter); end
        
        function self=configure( self, method, varsigma, mu, sigma )
        %
        % method: default 'tree'
        %   Method used for exploration step, either 'tree' or 'samp'.
        %
        % varsigma: default erfcinv(0.01)
        %   Expected probability that UCB < f.
        %   Another way to understand this parameter is that it controls how 
        %   "optimistic" we are during the exploration step. At a point x 
        %   evaluated using GP, the UCB will be: mu(x) + varsigma*sigma(x).
        %
        % mu: default 0
        %   Initial value of constant mean function.
        %
        % sigma: default 1e-3
        %   Initial std of Gaussian likelihood function (in normalised units).
        %
        % JH
            
            if nargin < 2, method = 'tree'; end
            if nargin < 3, varsigma = erfcinv(0.01); end 
            if nargin < 4, mu = 0; end
            if nargin < 5, sigma = 1e-3; end
            
            meanfunc = @meanConst; 
            covfunc  = {@covMaterniso, 5}; % isotropic Matern covariance 

            ell = 1/4; 
            sf  = 1;

            % hyper-parameters
            hyp.mean = mu; 
            hyp.lik  = log(sigma); 
            hyp.cov  = log([ell; sf]); 
            
            self.srgt.set_gp( hyp, meanfunc, covfunc );
            self.srgt.set_varsigma_const( varsigma );
            self.xmet = method;

        end
        
        function out = run( self, objfun, domain, Neval, varargin )
        %
        % out = run( self, objfun, domain, Neval, varargin )
        %
        % objfun:
        %   Function handle taking a candidate sample and returning a scalar.
        %   Expected input size should be 1 x Ndim.
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
        % InitSample: default L1-ball vertices
        %   Initial set of points to use for initialisation.
        %   Input can be an array of coordinates, in which case points are evaluated before optimisation.
        %   Or a structure with fields {coord,score}, in which case they are used directly by the surrogate.
        %
        % ExploreSize: default 5 if xmet='tree', 5*Ndim^2 otherwise
        %   Parameter for the exploration step (depth if xmet='tree', number of samples otherwise).
        %   You can set the exploration method manually (attribute xmet), or via the configure method.
        %   Note that in dimension D, you need a depth at least D if you want each dimension to be 
        %   split at least once. This becomes rapidly impractical as D increases, so you might want
        %   to select the sampling method instead if D is large.
        %
        % UpdateCycle: default 1
        %   Update constant for GP hyperparameters.
        %   See step_update for currently selected method.
        %
        % Verbose: default true
        %   Verbose switch.
        %
        % JH
        
            [objfun, domain, Neval, init, Xparam, upc, verbose] = ...
                self.checkargs( objfun, domain, Neval, varargin{:} );
            
            Ndim = size(domain,1);
            tstart = tic;
            self.verb = verbose;
            self.iter = {};
            
            % initialisation
            gpml_start();
            self.info( 'Start %d-dimensional optimisation, with budget of %d evaluations...', Ndim, Neval );
            [i_max,k_max] = self.initialise( domain, objfun, init );
            upn = self.step_update(upc,0);
            self.notify( 'PostInitialise' );
            
            % iterate
            while self.srgt.Ne < Neval
                
                self.info('\t------------------------------');
                self.notify( 'PreIteration' );
                
                self.step_explore(i_max,k_max,Xparam);
                [i_max,k_max] = self.step_select(objfun);
                upn = self.step_update(upc,upn);
                
                if nnz(i_max) == 0
                    warning( 'No leaf selected for iteration, aborting.' );
                    break;
                end
                
                self.progress(tstart,i_max);
                self.notify( 'PostIteration' );
                
            end
            gpml_stop();
            
            self.notify( 'PreFinalise' );
            out = self.finalise();
            
            self.info('');
            self.info('------------------------------');
            self.info('Best score out of %d samples: %g', numel(out.samp.f), out.sol.f);
            self.info('Total runtime: %s', dk.time.sec2str(toc(tstart)) );
            
        end
        
        function out = resume( self, objfun, Neval, varargin )
        %
        % out = resume( self, objfun, Neval, varargin )
        %
        % Resume optimisation, typically from unserialised GPSO object.
        % Note that the domain is not input here (extracted from surrogate instead).
        % You do need to provide the same objective function though, and any other option
        % set during the original run, to be consistent.
        %
        % JH
        
            domain = self.srgt.domain;
            [objfun, domain, Neval, ~, Xparam, upc, verbose] = ...
                self.checkargs( objfun, domain, Neval, varargin{:} );
            
            Ndim = size(domain,1);
            tstart = tic;
            self.verb = verbose;
            
            % initialisation
            gpml_start();
            self.info( 'Resume %d-dimensional optimisation, with budget of %d evaluations...', Ndim, Neval );
            Neval = Neval + self.srgt.Ne;
            i_max = self.iter{end}.split;
            k_max = self.get_k_max(i_max);
            upn = self.step_update(1,0); % force GP update
            
            % iterate
            while self.srgt.Ne < Neval
                
                self.info('\t------------------------------');
                self.notify( 'PreIteration' );
                
                self.step_explore(i_max,k_max,Xparam);
                [i_max,k_max] = self.step_select(objfun);
                upn = self.step_update(upc,upn);
                
                if nnz(i_max) == 0
                    warning( 'No leaf selected for iteration, aborting.' );
                    break;
                end
                
                self.progress(tstart,i_max);
                self.notify( 'PostIteration' );
                
            end
            gpml_stop();
            
            self.notify( 'PreFinalise' );
            out = self.finalise();
            
            self.info('');
            self.info('------------------------------');
            self.info('Best score out of %d samples: %g', numel(out.samp.f), out.sol.f);
            self.info('Total runtime: %s', dk.time.sec2str(toc(tstart)) );
        
        end
        
        function node = get_node(self,h,i)
        %
        % h: depth
        % i: depth-specific node index
        %
        % Get a node of the tree, with coordinates and sample info.
        %
        
            node = self.tree.node(h,i);
            node.coord = self.srgt.x(node.samp,:);
            node.samp  = self.srgt.y(node.samp,:);
        end
        
        function [best,S] = explore_tree(self,node,depth,varsigma)
        %
        % node: either a 1x2 array [h,i] (cf get_node), or a node structure
        % depth: how deep the exploration tree should be 
        %   (WARNING: tree grows exponentially!)
        % varsigma: optimism constant to be used locally for UCB
        %
        % Explore node by growing exhaustive partition tree, using surrogate for evaluation.
        %
            
            dk.assert( depth <= 8, [ ... 
                'This is safeguard error to prevent deep tree explorations.\n' ...
                'If you meant to set the option xmet="tree" with a depth of %d (%d samples),\n' ...
                'then please comment this message in the method explore_tree.\n' ...
            ], depth, 3^depth );
        
            % get node if index were passed
            if ~isstruct(node)
                node = self.get_node(node(1),node(2));
            end
            
            % number of points to sample
            S.coord = recursive_split( node, depth );
            
            % evaluate those points
            [mu,sigma] = self.srgt.gp_call( S.coord );
            ucb = mu + varsigma*sigma;
            
            [u,k]  = max(ucb);
            best   = [ mu(k), sigma(k), u ];
            S.samp = [ mu, sigma, ucb ];
            
        end
        
        function [best,S] = explore_samp(self,node,ns,varsigma)
        %
        % node: either a 1x2 array [h,i] (cf get_node), or a node structure
        % ns: number of random points to draw within the node
        % varsigma: optimism constant to be used locally for UCB
        %
        % Explore node with a uniformly random sample, using surrogate for evaluation. 
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
        
        % parse and verify inputs / options
        function [objfun, domain, Neval, init, Xparam, upc, vrb] ...
                = checkargs( self, objfun, domain, Neval, varargin )

            opt = dk.obj.kwArgs(varargin{:});
            
            assert( isa(objfun,'function_handle'), ...
                'Objective function should be a function handle.' );

            assert( ismatrix(domain) && ~isempty(domain) && ... 
                size(domain,2)==2 && all(diff(domain,1,2) > eps), 'Bad domain.' );
            
            lower = domain(:,1)';
            upper = domain(:,2)';

            Ndim = size(domain,1);
            Xdef = struct( 'tree', 5, 'samp', 5*Ndim^2 );
            Xdef = Xdef.(self.xmet);
            Idef = 0.5 + [ -0.25*eye(Ndim); +0.25*eye(Ndim) ]; % vertices of L1 ball of radius 0.25
            Idef = dk.bsx.add( lower, dk.bsx.mul(Idef,upper-lower) ); % denormalise 
        
            init   = opt.get( 'InitSample', Idef );
            Xparam = opt.get( 'ExploreSize', Xdef );
            upc    = opt.get( 'UpdateCycle', 1 );
            vrb    = opt.get( 'Verbose', true );

            if isstruct(init)
                assert( all(isfield( init, {'coord','score'} )), 'Missing initial sample field.' );
                assert( isnumeric(init.coord) && size(init.coord,2)==Ndim, 'Bad coord size.' );
                assert( isnumeric(init.score) && numel(init.score)==size(init.coord,1), 'Bad score size.' );
                Neval = Neval + size(init.coord,1); % don't count existing samples
                NeMin = 0;
            else
                assert( isnumeric(init) && size(init,2)==Ndim, 'Bad initial sample size.' );
                NeMin = size(init,1);
            end
            
            dk.assert( dk.is.integer(Neval) && Neval>NeMin, 'Neval should be >%d.', NeMin );
            dk.assert( dk.is.number(upc) && upc>0, 'upc should be >0.' );
            dk.assert( isscalar(vrb) && islogical(vrb), 'verbose should be boolean.' );

        end
        
        % keeping tabs on number of evaluated samples ..
        function upn=update_samp_linear(self,upc,upn)
            Ne = self.srgt.Ne;
            if (Ne-upn) >= upc

                self.info('\tHyperparameter update (neval=%d).',Ne);
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
                fprintf( ['[GPSO] ' fmt '\n'], varargin{:} );
            end
        end
        
        % print progress
        function data = progress(self,tstart,i_max)
            
            %data = [toc(tstart), self.tree.depth, Nselect, self.srgt.Ne, self.srgt.best_score];
            
            Ne = self.srgt.Ne;
            time = toc(tstart);
            best = self.srgt.best_score;
            data = struct( 'runtime', time, 'split', i_max, 'neval', Ne, 'score', best );
            
            self.iter{end+1} = data;
            self.info('\tEnd of iteration #%d (depth: %d, nselect: %d, neval: %d, score: %g)', ...
                self.Niter, numel(i_max), nnz(i_max), Ne, best );
            self.info('\t------------------------------ Elapsed time: %s\n', dk.time.sec2str(time) );
            
        end
        
        % get k_max from i_max
        function k_max = get_k_max(self,i_max)
            
            depth = self.tree.depth;
            k_max = zeros(1,depth);
            for h = 1:depth
                if i_max(h) > 0
                    k_max(h) = self.tree.samp(h,i_max(h));
                end
            end
            
        end
        
    end
    
    
    
    
        
    %% RUNTIME BREAKDOWN
    %
    %
    methods (Hidden,Access=private)
        
        function [i_max,k_max] = initialise(self,domain,objfun,init)
            
            % initialise surrogate
            self.srgt.init( domain );
            nd = self.srgt.Nd;
            
            % set initial points
            if ~isstruct(init)
                x = init;
                n = size(x,1);
                y = nan(n,1);
                for k = 1:n
                    y(k) = objfun(x(k,:));
                end
            else
                x = init.coord;
                n = size(x,1);
                y = init.score(:);
            end
            self.srgt.append( x, [y,zeros(n,1),y], false );
            
            % evaluate centre of the domain
            x = 0.5 + zeros(1,nd);
            y = objfun(self.srgt.denormalise(x));
            k = self.srgt.append( x, [y,0,y], true );
            
            % initialise tree
            self.tree.init(nd,k);
            
            % select root
            i_max = 1;
            k_max = k;
            
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
        
        % exploration step: split and sample
        function step_explore(self,i_max,k_max,Xparam)
            
            self.info('\tStep 1:');
            depth = self.tree.depth;
            varsigma = self.srgt.get_varsigma();
            
            Xfun = struct( 'tree', @self.explore_tree, 'samp', @self.explore_samp );
            Xfun = Xfun.(self.xmet);
            
            for h = 1:depth
            if i_max(h) > 0
                
                imax = i_max(h);
                kmax = k_max(h);
                
                % Split leaf along largest dimension
                [g,d,x,s] = split_largest_dimension( self.tree.level(h), imax, self.srgt.coord(kmax) );
                U = split_tree( self.tree.level(h), imax, g, d, x, s );
                Uget = @(n) struct( ...
                    'lower', U.lower(n,:), ...
                    'upper', U.upper(n,:), ...
                    'coord', U.coord(n,:)  ...
                );
                
                % Explore each new leaf with a uniform sample
                best_g = Xfun( Uget(1), Xparam, varsigma );
                best_d = Xfun( Uget(2), Xparam, varsigma );
                best_x = Xfun( Uget(3), Xparam, varsigma );
                
                % Append points and update tree
                k = self.srgt.append( [g;d;x], [best_g;best_d;best_x], true );
                self.tree.split( h, imax, U.lower, U.upper, k );
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

                % find leaf node with score greater than any elder leaf node
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
        
        % facade method for hyperparameter update
        function upn = step_update(self,upc,upn)
            upn = self.update_samp_linear(upc,upn);
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

function coord = recursive_split(node,k)

    Tmin = node.lower;
    Tmax = node.upper;
    
    x = node.coord;
    if k == 0 % termination clause
        coord = x;
        return; 
    end
    
    g = x;
    d = x;
    
    [~,s] = max( Tmax - Tmin );
    g(s)  = (5*Tmin(s) +   Tmax(s))/6;
    d(s)  = (  Tmin(s) + 5*Tmax(s))/6;
    
    Gmax = Tmax;
    Dmin = Tmin;
    Xmin = Tmin;
    Xmax = Tmax;
    
    Gmax(s) = (2*Tmin(s) +   Tmax(s))/3.0;
    Dmin(s) = (  Tmin(s) + 2*Tmax(s))/3.0;
    Xmin(s) = Gmax(s);
    Xmax(s) = Dmin(s);
    
    % recursion
    make_node = @(ll,uu,xx) struct('lower',ll,'upper',uu,'coord',xx);
    coord = [ ...
        recursive_split(make_node(Tmin,Gmax,g),k-1);
        recursive_split(make_node(Dmin,Tmax,d),k-1);
        recursive_split(make_node(Xmin,Xmax,x),k-1)
    ];

end

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
    
    % careful, the order matters
    U.coord = [g;d;x];
    U.lower = [Tmin;Dmin;Xmin];
    U.upper = [Gmax;Tmax;Xmax];

end
