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
        
        % serialise data to be saved
        function D = serialise(self)
            F = {'level','Nl','Ns'};
            n = numel(F);
            D = struct();
            
            for i = 1:n
                f = F{i};
                D.(f) = self.(f);
            end
            D.version = '0.2';
        end
        function self=unserialise(self,D)
            F = {'level','Nl','Ns'};
            n = numel(F);
            
            for i = 1:n
                f = F{i};
                self.(f) = D.(f);
            end
            
            switch D.version
                case '0.1'
                    % used to store level and index, but level is useless since it's always the one just above
                    for h = 1:self.depth
                        assert( all( self.level(h).parent(:,1) == h-1 ), 'Bad ancestry record.' );
                        self.level(h).parent = self.level(h).parent(:,2)';
                    end
                    if self.depth
                        self.level(1).parent = 0; % root index is 0
                    end
            end
        end
        
        function self=init(self,ndim,rid)
        %
        % ndim: dimensionality of search space
        % rid: storage index of tree root (centre of hyperdomain)
        % 
            
            if nargin < 3, rid=1; end
            
            % initialise tree
            T.parent = 0;
            T.lower  = zeros(1,ndim);
            T.upper  = ones(1,ndim);
            T.samp   = rid;
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
            p = self.level(h).parent(k);
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
        
        function self=split(self,h,k,l,u,s)
        % 
        % h: parent level
        % k: parent id 
        % l: nxd array of lower bound(s)
        % u: nxd array of upper bound(s)
        % s: nx1 sample id of the surrogate
        %
        
            assert( self.leaf(h,k), '[bug] Splitting non-leaf node.' );
            self.level(h).leaf(k) = false;
            
            n = numel(s);
            m = h+1;
            
            if m > self.depth
                self.level(m).leaf = true(0); % initialise next level
            end
            
            self.level(m).parent = [self.level(m).parent, k*ones(1,n)];
            self.level(m).lower  = [self.level(m).lower; l];
            self.level(m).upper  = [self.level(m).upper; u];
            self.level(m).samp   = [self.level(m).samp, s];
            self.level(m).leaf   = [self.level(m).leaf, true(1,n)];
            
            self.Ns = self.Ns+1;
            self.Nl = self.Nl+n-1;
            
        end
        
        function T = export_compact(self)
        %
        % Export tree as a 5xn array where rows correspond to:
        %
        %     parent_id  Index of the parent node in this array.
        %   children_id  Zero for leaf-nodes, otherwise index of first child.
        %     sample_id  Index of the corresponding surrogate sample.
        %         depth  Depth of the node in the tree.
        %         order  Index of seniority (higher values mean younger).
        %
        % Note: children are necessarily next to each other, so if C is the index of the first child,
        %   then the other children are C+1 and C+2.
        % 
        
            d = self.depth; % tree depth
            w = arrayfun( @(x) numel(x.samp), self.level ); % width of each level
            n = sum(w); % total number of nodes
            
            % first pass:
            %   set depth, sample indices, and re-index parents
            T = zeros(1,n);
            T = struct( 'parent', T, 'children', T, 'sample', T, 'depth', T, 'order', T, 'n', n, 'd', d );
            b = 1;
            e = 0;
            for h = 1:d
                o = b-1;
                b = e+1;
                e = b+w(h)-1;
                
                T.parent(b:e) = o + self.level(h).parent;
                T.sample(b:e) = self.level(h).samp;
                T.depth(b:e) = h;
            end
            
            % second pass: set children 
            for i = 1:n
                p = T.parent(i); 
                if p>0 && T.children(p)==0
                    T.children(p) = i;
                end
            end
            
            % third pass: set order
            for i = 2:3:n
                k = i + [0 1 2];
                T.order(k) = max(T.sample(k));
            end
            T.order(1) = T.sample(1); % set the root
            
            % re-index the order
            u = unique(T.order);
            r(u) = 1:numel(u);
            T.order = r(T.order);
            
        end
        
    end
    
end