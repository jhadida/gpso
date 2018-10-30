classdef GPSO < handle
%
% Free software provided under AGPLv3 license (see README).
% Copyright Jonathan Hadida (jhadida@fmrib.ox.ac.uk), July 2017.

    properties
        verb % verbose switch (true/false)
    end
    
    properties (SetAccess=private)
        srgt % GP surrogate
        tree % partition tree
        iter % iteration data
    end
    
    properties (Transient,Dependent)
        Niter
    end
    
    % use events to trigger processing/monitoring functions during optimisation
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
        end
        
        % dependent parameter to get the current iteration count
        % useful when working with event callbacks
        function n = get.Niter(self), n=numel(self.iter); end
        
        function self=configure( self, varsigma, mu, sigma )
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
            
            if nargin < 2, varsigma = erfcinv(0.01); end 
            if nargin < 3, mu = 0; end
            if nargin < 4, sigma = 1e-3; end
            
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

        end
        
        function out = run( self, objfun, domain, Neval, Xmet, Xprm, varargin )
        %
        % out = run( self, objfun, domain, Neval, Xmet, Xprm, varargin )
        %
        % objfun:
        %   Function handle taking a candidate sample and returning a scalar.
        %   Expected input size should be 1 x Ndim.
        %   The optimisation MAXIMISES this function.
        %
        % domain:
        %   Ndim x 2 matrix specifying the boundaries of the cartesian domain.
        %
        % Neval:
        %   Maximum number of function evaluation.
        %   This can be considered as a "budget" for the optimisation.
        %   Note that the actual number of evaluations can exceed this value (usually not by much).
        %
        % Xmet: default 'tree'
        %   Method used for exploration of children intervals.
        %     - tree explores by recursively applying the partition function;
        %     - samp explores by randomly sampling the GP within the interval.
        %
        % Xprm: default 5 if xmet='tree', 5*Ndim^2 otherwise
        %   Parameter for the exploration step (depth if xmet='tree', number of samples otherwise).
        %   You can set the exploration method manually (attribute xmet), or via the configure method.
        %   Note that in dimension D, you need a depth at least D if you want each dimension to be 
        %   split at least once. This becomes rapidly impractical as D increases, so you might want
        %   to select the sampling method instead if D is large.
        %
        % KEY/VALUE OPTIONS:
        %
        % InitSample:  by default, two vertices per dimension, equally spaced from the centre (diamond-shape).
        %   Initial set of points to use for initialisation.
        %   Input can be an array of coordinates, in which case points are evaluated before optimisation.
        %   Or a structure with fields {coord,score}, in which case they are used directly by the surrogate.
        %
        % UpdateCycle: by default, update at each iteration.
        %   Update constant for GP hyperparameters.
        %   See step_update for currently selected method.
        %
        % Verbose: default true
        %   Verbose switch.
        %
        % JH
        
            [objfun, domain, Neval, init, Xplore, upc, verbose] = ...
                checkargs( objfun, domain, Neval, Xmet, Xprm, varargin{:} );
            
            Ndim = size(domain,1);
            tstart = tic;
            self.verb = verbose;
            self.iter = {};
            
            % initialisation
            gpml_start();
            self.info( 'Start %d-dimensional optimisation, with budget of %d evaluations...', Ndim, Neval );
            i_max = self.initialise( domain, objfun, init );
            upn = self.step_update(upc,0);
            self.notify( 'PostInitialise' );
            
            % iterate
            while self.srgt.Ne < Neval
                
                self.info('\t------------------------------');
                self.notify( 'PreIteration' );
                
                self.step_explore(i_max,Xplore);
                i_max = self.step_select(objfun);
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
        
        function out = resume( self, objfun, Neval, Xmet, Xprm, varargin )
        %
        % out = resume( self, objfun, Neval, Xmet, Xprm, varargin )
        %
        % Resume optimisation, typically from unserialised GPSO object.
        % See run method above for description of inputs and options (InitSample ignored).
        %
        % NOTE: the domain is not input here (extracted from the surrogate instead).
        % You DO need to provide the same objective function though, and the same exploration
        % options set during the original run, for consistency.
        %
        % JH
        
            domain = self.srgt.domain;
            [objfun, domain, ~, ~, Xplore, upc, verbose] = ...
                checkargs( objfun, domain, Neval, Xmet, Xprm, varargin{:} );
            % Ignore Neval output because we ignore InitSample and that's the only way it can change
            
            Ndim = size(domain,1);
            tstart = tic;
            self.verb = verbose;
            
            % no real initialisation, just use the surrogate in its current state
            gpml_start();
            self.info( 'Resume %d-dimensional optimisation, with budget of %d evaluations...', Ndim, Neval );
            Neval = Neval + self.srgt.Ne;
            upn = self.step_update(1,0); % force GP update
            skipexp = true; % skip first exploration
            
            % iterate
            while self.srgt.Ne < Neval
                
                self.info('\t------------------------------');
                self.notify( 'PreIteration' );
                
                if skipexp
                    skipexp = false; 
                else
                    self.step_explore(i_max,Xplore);
                end
                i_max = self.step_select(objfun);
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
        
        % serialise data to be saved
        function D = serialise(self,filename)
            D.iter = self.iter;
            D.tree = self.tree.serialise();
            D.surrogate = self.srgt.serialise();
            D.version = '0.1';
            
            if nargin > 1
                self.info( 'Serialised into file: %s', filename );
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
                upn = dk.num.nextint( Nsplit/upc );
                self.notify( 'PostUpdate' );

            end
        end
        
        function upn=update_split_quadratic(self,upc,upn)
            Nsplit = self.tree.Ns;
            if 2*Nsplit >= upc*upn*(upn+1)

                self.info('\tHyperparameter update (nsplit=%d).',upn);
                self.srgt.gp_update();
                upn = dk.num.nextint( (sqrt(1+8*Nsplit/upc)-1)/2 );
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
        function k_max = imax2kmax(self,i_max)
            
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
    % The different steps of the algorithm.
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
                for i = 1:n
                    y(i) = objfun(x(i,:));
                end
            else
                self.info('Using user-specified initial sample.');
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
            
            % select root for split
            i_max = 1;
            k_max = k; % corresponding surrogate index
            
        end
        
        function out = finalise(self)
            
            % list all evaluated samples
            [x,f] = self.srgt.samp_evaluated(true);
            out.samp.x = x;
            out.samp.f = f; % associated scores
            
            % get best sample
            [x,f] = self.srgt.best_sample(true);
            out.sol.x = x;
            out.sol.f = f;
            
        end

        % exploration step: split and sample
        function step_explore(self,i_max,xobj)
            
            self.info('\tStep 1:');
            depth = self.tree.depth;
            
            for h = 1:depth
            if i_max(h) > 0
                
                self.tree.split( h, i_max(h), self.srgt, xobj.method, xobj.param );
                self.info('\t\t[h=%02d] Split leaf %d',h,i_max(h));
                
            end % if
            end % for
            
        end

        % selection step: select and evaluate
        function [i_max,k_max] = step_select(self,objfun)
            
            self.info('\tStep 2:');
            depth = self.tree.depth;
            i_max = zeros(depth,1); % tree indices
            k_max = zeros(depth,1); % corresponding surrogate indices
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

                % if a leaf is selected, and that it is gp-based, evaluate it
                imax = i_max(h);
                kmax = k_max(h);
                if (imax > 0) && self.srgt.is_gp_based(kmax)
                    self.info('\t\t[h=%02d] Sampling GP-based leaf %d with UCB %g',h,imax,v_max);
                    f = objfun(self.srgt.coord(kmax,true));
                    self.srgt.update( kmax, [f,0,f] ); % Note: important NOT to keep v_max here
                end

                if imax > 0
                    self.info('\t\t[h=%02d] Select leaf %d with score %g',h,imax,v_max);
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





% parse and verify inputs / options
function [objfun, domain, Neval, init, Xplore, upc, vrb] ...
        = checkargs( objfun, domain, Neval, Xmet, Xprm, varargin )

    % objective
    assert( isa(objfun,'function_handle'), ...
        'Objective function should be a function handle.' );

    % domain
    assert( ismatrix(domain) && ~isempty(domain) && ... 
        size(domain,2)==2 && all(diff(domain,1,2) > eps), 'Bad domain.' );

    lower = domain(:,1)';
    upper = domain(:,2)';
    Ndim  = size(domain,1);
    
    % exploration
    Xdef = struct( 'tree', 5, 'samp', 5*Ndim^2 );
    if nargin < 4 || isempty(Xmet), Xmet = 'samp'; end
    if nargin < 5 || isempty(Xprm), Xprm = Xdef.(Xmet); end
    Xplore.method = Xmet;
    Xplore.param = Xprm;
    
    % other inputs
    Idef = 0.5 + [ -0.25*eye(Ndim); +0.25*eye(Ndim) ]; % vertices of L1 ball of radius 0.25
    Idef = dk.bsx.add( lower, dk.bsx.mul(Idef,upper-lower) ); % denormalise 
    
    opt  = dk.obj.kwArgs(varargin{:});
    init = opt.get( 'InitSample', Idef );
    upc  = opt.get( 'UpdateCycle', 1 );
    vrb  = opt.get( 'Verbose', true );

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

    dk.assert( dk.is.number(Neval) && Neval>NeMin, 'Neval should be >%d.', NeMin );
    dk.assert( dk.is.number(upc) && upc>0, 'upc should be >0.' );
    dk.assert( isscalar(vrb) && islogical(vrb), 'verbose should be boolean.' );

end



% % DOESNT WORK FOR NOW
% %
% function NNS = reinitialise(self,domain,init,samp)
% 
%     % initialise surrogate
%     self.srgt.init( domain );
%     nd = self.srgt.Nd;
% 
%     % normalise sample
%     samp.coord = self.srgt.normalise(samp.coord);
% 
%     % train GP
%     self.srgt.gp_update( samp.coord, samp.score );
% 
%     % create searcher
%     NNS = dk.obj.GeoData_NNS( samp.coord, samp.score );
% 
%     % set initial points
%     if ~isstruct(init)
%         x = self.srgt.normalise(init);
%         n = size(x,1);
%         y = nan(n,1);
%         for k = 1:n
%             y(k) = NNS.get_data(x(k,:));
%         end
%     else
%         x = self.srgt.normalise(init.coord);
%         n = size(x,1);
%         y = init.score(:);
%     end
%     self.srgt.append( x, [y,zeros(n,1),y], true );
% 
%     % find centre of the domain
%     x = 0.5 + zeros(1,nd);
%     y = NNS.get_data(x);
%     k = self.srgt.append( x, [y,0,y], true );
% 
%     % initialise tree
%     self.tree.init(nd,k);
% 
% end
% 
% function self = rebuild( self, samp, maxdepth, domain, Xmet, Xprm, varargin )
% %
% % self = rebuild( self, samp, maxdepth, domain, Xmet, Xprm, varargin )
% %
% % Rebuild optimiser state from input sample.
% %
% % samp must be a structure with fields {coord,score}, which should contain initial 
% %   points as well as points sampled during optimisation.
% % maxdepth is there to limit the search for sampled points in children intervals,
% %   mainly because the middle child will always be found due to the recursive nature
% %   of ternary splits.
% %
% % domain, Xmet and Xprm, and other options are as usual (see method run).
% % UpdateCycle is ignored.
% %
% % JH
% 
%     % check sample and maxdepth
%     assert( isstruct(samp) && all(isfield(samp,{'coord','score'})), 'Bad sample.' );
%     Nsamp = numel(samp.score);
% 
%     assert( size(samp.coord,1) == Nsamp, 'Sample size mismatch.' );
%     assert( dk.is.number(maxdepth) && maxdepth >= log(Nsamp)/log(3), 'Bad depth.' );
% 
%     % parse other inputs
%     [~, domain, ~, init, Xplore, ~, verbose] = ...
%         checkargs( @(x)x, domain, Inf, Xmet, Xprm, varargin{:} );
%     self.verb = verbose;
% 
%     % re-initialise surrogate
%     NNS = self.reinitialise( domain, init, samp );
% 
%     % exploration parameters
%     varsigma = self.srgt.get_varsigma();
% 
%     Xmet = Xplore.method;
%     Xprm = Xplore.param;
%     Xfun = struct( 'tree', @self.explore_tree, 'samp', @self.explore_samp );
%     Xfun = Xfun.(Xmet);
% 
%     % recursive DFS with feedback
%     function yes = should_split(pid,node,h)
%         test = @(k) (k > 0) && (k ~= pid);
% 
%         if h == maxdepth
%             yes = test(NNS.find(node.coord));
%         else
%             child = recursive_split( node, 1 );
%             yes = test(NNS.find(child(1).coord)) ...
%                 || test(NNS.find(child(2).coord)) ...
%                 || test(NNS.find(child(3).coord)) ...
%                 || should_split(pid,child(1),h+1) ...
%                 || should_split(pid,child(2),h+1) ...
%                 || should_split(pid,child(3),h+1);
%         end
%     end
% 
%     function s = get_score(node)
%         try
%             s = NNS.get_data(node.coord);
%             s = [s,0,s];
%         catch
%             s = Xfun( node, Xprm, varsigma );
%         end
%     end
% 
%     for h = 1:maxdepth
% 
%         w = self.tree.width(h);
%         for i = 1:w
%             n = self.get_node(h,i);
%             k = NNS.find(n.coord);
%             if (k > 0) && should_split(k,n,h)
% 
%                 % Split leaf along largest dimension
%                 [g,d,x,s] = split_largest_dimension( self.tree.level(h), i, n.coord );
%                 U = split_tree( self.tree.level(h), i, g, d, x, s );
%                 Uget = @(j) struct( ...
%                     'lower', U.lower(j,:), ...
%                     'upper', U.upper(j,:), ...
%                     'coord', U.coord(j,:)  ...
%                 );
% 
%                 % Explore each new leaf with a uniform sample
%                 best_g = get_score(Uget(1));
%                 best_d = get_score(Uget(2));
%                 best_x = get_score(Uget(3));
% 
%                 % Append points and update tree
%                 k = self.srgt.append( [g;d;x], [best_g;best_d;best_x], true );
%                 self.tree.split( h, i, U.lower, U.upper, k );
% 
%             end
%         end
% 
%         % early canceling
%         if all(NNS.access), break; end
% 
%     end
% 
%     % check that all points are found
%     assert( all(NNS.access), 'Some points were not found.' );
% 
% end
