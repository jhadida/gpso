classdef GPSO_Tree < handle
    
    properties (SetAccess = private)
        level;
        Nl, Ns; % number of leaves/splits
    end
    
    properties (Transient,Dependent)
        depth;
    end
    
    methods
        
        function self = GPSO_Tree()
            self.clear();
        end
        
        function self=clear(self)
            self.level = [];
            self.Nl = 0;
            self.Ns = 0;
        end
        
        function self=init(self,ndim)
            
            % initialise tree
            T.parent = [0,1];
            T.lower  = zeros(1,ndim);
            T.upper  = ones(1,ndim);
            T.samp   = 1;
            T.leaf   = true;
            
            self.level = T;
            self.Nl = 1;
            self.Ns = 0;
            
        end
        
        function d = get.depth(self)
            d = numel(self.level);
        end
        function w = width(self,h)
            w = numel(self.level(h).samp);
        end
        
        % quick access
        function p = parent(self,h,k)
            p = self.level(h).parent(k,:);
        end
        function l = lower(self,h,k)
            l = self.level(h).lower(k,:);
        end
        function u = upper(self,h,k)
            u = self.level(h).upper(k,:);
        end
        function s = samp(self,h,k)
            s = self.level(h).samp(k);
        end
        function l = leaf(self,h,k)
            l = self.level(h).leaf(k);
        end
        
        % works with k vector, but h should be scalar
        % not super efficient, don't use too often
        function n = node(self,h,k)
            n.parent = self.parent(h,k);
            n.lower  = self.lower(h,k);
            n.upper  = self.upper(h,k);
            n.samp   = self.samp(h,k);
            n.leaf   = self.leaf(h,k);
        end
        
        function self=split(self,p,l,u,s)
        % 
        % p: [parent_level,parent_id]
        % l: nxd array of lower bound(s)
        % u: nxd array of upper bound(s)
        % s: nx1 sample id of the surrogate
        %
            
            assert( numel(p)==2, 'Splitting should involve only one parent node.' );
            
            n = numel(s);
            h = p(1);
            k = p(2);
            m = h+1;
            
            assert( self.leaf(h,k), '[bug] Splitting non-leaf node.' );
            self.level(h).leaf(k) = false;
            
            if m > self.depth
                self.level(m).leaf = true(0); % initialise next level
            end
            
            self.level(m).parent = [self.level(m).parent; repmat(p,n,1)];
            self.level(m).lower  = [self.level(m).lower; l];
            self.level(m).upper  = [self.level(m).upper; u];
            self.level(m).samp   = [self.level(m).samp, s];
            self.level(m).leaf   = [self.level(m).leaf, true(1,n)];
            
            self.Ns = self.Ns+1;
            self.Nl = self.Nl+n-1;
            
        end
        
        % serialise data to be saved
        function D = serialise(self)
            F = {'level','Nl','Ns'};
            n = numel(F);
            D = struct();
            
            for i = 1:n
                f = F{i};
                D.(f) = self.(f);
            end
            D.version = '0.1';
        end
        function self=unserialise(self,D)
            F = {'level','Nl','Ns'};
            n = numel(F);
            
            for i = 1:n
                f = F{i};
                self.(f) = D.(f);
            end
        end
        
    end
    
end