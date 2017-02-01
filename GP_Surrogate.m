classdef GP_Surrogate < handle
    
    properties (SetAccess = private)
        x; % sample coordinates
        y; % data array with columns [mu,sigma,ucb]
        GP; % GP parameters
        
        lower;
        upper;
        
        Ne, Ng; % number of evaluated/GP-based samples
    end
    
    properties (Transient,Dependent)
        Ns; % number of samples
        Nd; % number of dimensions
        delta; % dimensions of the box
    end
    
    methods
        
        function self=GP_Surrogate()
            self.clear();
        end
        
        function self=clear(self)
            self.x = [];
            self.y = [];
            self.GP = struct();
            self.lower = [];
            self.upper = [];
            self.Ne = 0;
            self.Ng = 0;
        end
        
        function self=gpconf(self, hyp, meanfunc, covfunc, eta)
            
            if nargin < 5, eta=0.05; end % probability that UCB < f 
            
            gp.hyp = hyp;
            gp.likfunc = @likGauss; % Gaussian likelihood is assumed for analytics
            gp.meanfunc = meanfunc;
            gp.covfunc = covfunc;
            gp.varsigma = @(M) sqrt(max( 0, 4*log(pi*M) - 2*log(12*eta) )); % cf Lemma 1
            % JH: original varsigma can be complex for M=1 and eta=1
            
            self.GP = gp;
            
        end
        
        % WARNING: by default, assumes x is NOT normalised
        function self=init(self,domain)
            
            assert( ismatrix(domain) && ~isempty(domain) && ... 
                size(domain,2)==2 && all(diff(domain,1,2) > eps), 'Bad domain.' );
        
            self.lower = domain(:,1)';
            self.upper = domain(:,2)';
            
            self.x = [];
            self.y = [];
            
            self.Ne = 0;
            self.Ng = 0;
            
        end
                
        % normalise/denormalise coordinates
        function y = normalise(self,x)
            y = bsxfun( @minus, x, self.lower );
            y = bsxfun( @rdivide, y, self.delta );
        end
        function y = denormalise(self,x)
            y = bsxfun( @times, x, self.delta );
            y = bsxfun( @plus, y, self.lower );
        end
        
        % append new samples
        % WARNING: by default, assumes x is NORMALISED
        function k=append(self,x,m,s,isnorm)
            
            n = numel(m);
            if nargin < 4, s=zeros(n,1); end
            if nargin < 5, isnorm=true; end
            
            if isscalar(s) && n>1, s=s*ones(n,1); end
            assert( size(x,1)==n && numel(s)==n, 'Size mismatch.' );
            assert( all(s >= 0), 'Sigma should be non-negative.' );
            
            k = self.Ns + (1:n);
            if ~isnorm, x = self.normalise(x); end
            
            m = m(:);
            s = s(:);
            
            self.x = [self.x; x ];
            self.y = [self.y; [m,s,m] ];
            
            g = nnz(s);
            self.Ng = self.Ng + g;
            self.Ne = self.Ne + n-g;
            
        end
        
        % evaluate GP-based sample
        function self=edit(self,k,f)
            
            n = numel(k);
            assert( numel(f)==n, 'Size mismatch.' );
            f = f(:);
            
            assert( all(self.sigma(k) > 0), '[bug] Updating non GP-based sample.' );
            self.y(k,:) = [f,zeros(n,1),f];
            
            self.Ne = self.Ne + n;
            self.Ng = self.Ng - n;
            
        end
        
        % evaluate surrogate at query points
        % WARNING: assumes query points xq are NOT normalised
        function [m,s] = surrogate(self,xq)
            [m,s] = self.gp_call(self.normalise(xq));
        end
        
        % serialise data to be saved
        function D = serialise(self)
            F = {'lower','upper','x','y','Ne','Ng','GP'};
            n = numel(F);
            D = struct();
            
            for i = 1:n
                f = F{i};
                D.(f) = self.(f);
            end
            D.version = '0.1';
        end
        function self=unserialise(self,D)
            F = {'lower','upper','x','y','Ne','Ng','GP'};
            n = numel(F);
            
            for i = 1:n
                f = F{i};
                self.(f) = D.(f);
            end
        end
        
    end
    
    methods
        
        % dependent properties
        function n = get.Ns(self), n = size(self.x,1); end
        function n = get.Nd(self), n = size(self.x,2); end
        function d = get.delta(self), d = self.upper-self.lower; end
        
        % named access to y's columns
        function y = ycol(self,c,k)
            if nargin > 2
                y = self.y(k,c);
            else
                y = self.y(:,c);
            end
        end
        
        function m = mu(self,varargin), m = self.ycol(1,varargin{:}); end
        function s = sigma(self,varargin), s = self.ycol(2,varargin{:}); end
        function u = ucb(self,varargin), u = self.ycol(3,varargin{:}); end
        
        % access sample coordinates
        function x = coord(self,k,denorm)
            if nargin < 3, denorm=false; end
            x = self.x(k,:);
            if denorm
                x = self.denormalise(x);
            end
        end
        
        % is GP-based
        function g = gp_based(self,varargin)
            g = self.sigma(varargin{:}) > 0;
        end
        
        % indices of evaluated/gp-based samples
        function k = find_evaluated(self)
            k = find( self.sigma == 0 );
        end
        function k = find_gp_based(self)
            k = find( self.sigma > 0 );
        end
        
        % access evaluated/gp-based samples
        function [x,f] = samp_evaluated(self,varargin)
            k = self.find_evaluated();
            f = self.ucb(k);
            x = self.coord(k,varargin{:});
        end
        function [x,f] = samp_gp_based(self,varargin)
            k = self.find_gp_based();
            f = self.ucb(k);
            x = self.coord(k,varargin{:});
        end
        
        % best score or sample
        function [f,k] = best_score(self)
            p = self.find_evaluated();
            [f,k] = max(self.ucb(p));
            k = p(k);
        end
        function [x,f] = best_sample(self,varargin)
            [f,k] = self.best_score();
            x = self.coord(k,varargin{:});
        end
        
    end
    
    methods
        
        % UCB update
        function self=ucb_update(self)
            if self.Ng > 0
                self.y(:,3) = self.y(:,1) + self.GP.varsigma(self.Ng) * self.y(:,2);
            end
        end
        
        % check a few things before calling gp
        function gp_check(self)

            % make sure GP is set
            assert( isfield(self.GP,'varsigma'), 'GP has not been configured yet (see method gpconf).' );
            
            % make sure GPML is on the path
            if isempty(which('gp'))
                warning('GPML library not on the path, calling gpml_start.');
                gpml_start();
            end
            
            % don't allow sigma to become too small
            self.GP.hyp.lik = max( self.GP.hyp.lik, -15 );
        
        end
        
        % WARNING: assumes query points are NORMALISED
        function [m,s]=gp_call(self,xq)
            
            self.gp_check();
            
            err = true;
            hyp = self.GP.hyp;
            
            [xe,fe] = self.samp_evaluated();
            while err
                err = false;
                try
                    [m,s] = gp( hyp, @infExact, ...
                        self.GP.meanfunc, self.GP.covfunc, self.GP.likfunc, xe, fe, xq ...
                    );
                catch
                    err = true;
                    hyp.lik = hyp.lik + 1;
                end
            end
            s = sqrt(s); % gp returns variance
            
        end
        
        function self=gp_update(self)
            
            self.gp_check();
            
            [xe,fe] = self.samp_evaluated();
            self.GP.hyp = minimize( self.GP.hyp, @gp, -100, ...
                @infExact, self.GP.meanfunc, self.GP.covfunc, self.GP.likfunc, xe, fe );
            
            % don't allow sigma to become too small
            self.GP.hyp.lik = max( self.GP.hyp.lik, -15 );
            
        end
        
    end
    
end